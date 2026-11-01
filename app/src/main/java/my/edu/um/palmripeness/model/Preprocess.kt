package my.edu.um.palmripeness.model

import kotlin.math.ceil
import kotlin.math.floor

/**
 * The training pipeline's preprocessing, reimplemented in Kotlin.
 *
 * This is the part of the app that is easy to get silently wrong: feed [0, 255] or [0, 1]
 * instead of [-1, 1], centre-crop instead of squashing, apply ImageNet mean/std a second time
 * (FastViT's normalisation is inside the graph), and the model still returns four confident
 * softmax scores. `app/src/main/assets/model_contract.json` is the written contract and
 * `app/src/test/resources/resize_parity.json` is the test that this matches `tf.image.resize`.
 *
 * Deliberately free of Android types so the parity test runs as a plain JVM unit test.
 */
object Preprocess {
    const val SIZE = 224

    /** Supplies one row of the source image as RGB triples, each value 0..255. */
    fun interface RowSource {
        /** Fills [out] (length `srcWidth * 3`) with row [y] of the source image. */
        fun row(y: Int, out: FloatArray)
    }

    /**
     * Bilinear resize matching `tf.image.resize(..., method='bilinear')` with half-pixel
     * centres and no antialiasing: the whole frame is squashed to [target] x [target], aspect
     * ratio not preserved, no crop.
     *
     * Bilinear interpolation is separable, so resizing each needed source row across and then
     * interpolating down between two of them is the same arithmetic as TensorFlow's four-corner
     * form — which matters because it means only two source rows are ever in memory, and a
     * 12 MP gallery photo does not have to be materialised as floats.
     *
     * @return `target * target * 3` floats, HWC, still in [0, 255].
     */
    fun resizeBilinear(
        srcHeight: Int,
        srcWidth: Int,
        target: Int = SIZE,
        source: RowSource,
    ): FloatArray {
        require(srcHeight > 0 && srcWidth > 0) { "empty source image ${srcWidth}x$srcHeight" }
        val x = weights(srcWidth, target)
        val y = weights(srcHeight, target)

        val srcRow = FloatArray(srcWidth * 3)
        val rowWidth = target * 3
        // Two x-resized source rows, kept with the source y they were read from, because
        // consecutive output rows almost always reuse one of them.
        var cachedTopY = -1
        var cachedBottomY = -1
        var cachedTop = FloatArray(rowWidth)
        var cachedBottom = FloatArray(rowWidth)

        val out = FloatArray(target * target * 3)
        for (oy in 0 until target) {
            val topY = y.lower[oy]
            val bottomY = y.upper[oy]

            if (cachedBottomY == topY) {
                // The previous output row's bottom is this one's top: swap instead of re-reading.
                val swap = cachedTop; cachedTop = cachedBottom; cachedBottom = swap
                cachedTopY = topY; cachedBottomY = -1
            }
            if (cachedTopY != topY) {
                source.row(topY, srcRow)
                resizeRowAcross(srcRow, x, cachedTop)
                cachedTopY = topY
            }
            if (cachedBottomY != bottomY) {
                if (bottomY == topY) {
                    cachedTop.copyInto(cachedBottom)
                } else {
                    source.row(bottomY, srcRow)
                    resizeRowAcross(srcRow, x, cachedBottom)
                }
                cachedBottomY = bottomY
            }

            val yLerp = y.lerp[oy]
            val base = oy * rowWidth
            for (i in 0 until rowWidth) {
                val top = cachedTop[i]
                out[base + i] = top + (cachedBottom[i] - top) * yLerp
            }
        }
        return out
    }

    /** `x / 127.5 - 1`, applied in place to values in [0, 255]. */
    fun scaleToModelRange(pixels: FloatArray): FloatArray {
        for (i in pixels.indices) pixels[i] = pixels[i] / 127.5f - 1f
        return pixels
    }

    private fun resizeRowAcross(srcRow: FloatArray, x: Weights, out: FloatArray) {
        for (ox in 0 until x.lower.size) {
            val lo = x.lower[ox] * 3
            val hi = x.upper[ox] * 3
            val lerp = x.lerp[ox]
            val o = ox * 3
            for (c in 0..2) {
                val a = srcRow[lo + c]
                out[o + c] = a + (srcRow[hi + c] - a) * lerp
            }
        }
    }

    private class Weights(val lower: IntArray, val upper: IntArray, val lerp: FloatArray)

    /**
     * TensorFlow's `compute_interpolation_weights`, half-pixel centres. Note that `lerp` uses
     * the unclamped floor while `lower` is clamped at zero: for the first output pixel of an
     * upscale the two bounds collapse to the same source pixel, so the weight is irrelevant,
     * and mirroring TF here is cheaper than reasoning about when it is not.
     */
    private fun weights(inSize: Int, outSize: Int): Weights {
        val scale = inSize.toFloat() / outSize.toFloat()
        val lower = IntArray(outSize)
        val upper = IntArray(outSize)
        val lerp = FloatArray(outSize)
        for (i in 0 until outSize) {
            val inF = (i + 0.5f) * scale - 0.5f
            val fl = floor(inF)
            lower[i] = maxOf(fl.toInt(), 0)
            upper[i] = minOf(ceil(inF).toInt(), inSize - 1)
            lerp[i] = inF - fl
        }
        return Weights(lower, upper, lerp)
    }
}
