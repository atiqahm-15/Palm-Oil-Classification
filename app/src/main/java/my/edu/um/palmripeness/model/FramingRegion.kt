package my.edu.um.palmripeness.model

import kotlin.math.roundToInt

/**
 * Where the camera's framing brackets sat, as a fraction of the preview, so the captured frame can
 * be cut down to what the user actually framed before it reaches [Preprocess].
 *
 * Why this exists. Every image the model was trained on is a tight crop around one bunch. Handing
 * it the whole previewed frame instead — a bunch inside the brackets, surrounded by whatever else
 * the phone could see — costs 45.7 points of test accuracy under simulation
 * (`results/framing_shift/`, and §8 of `PROJECT_PLAN.md`). Cropping to the brackets is what puts
 * the input back in the distribution the model knows.
 *
 * Deliberately free of Android types so the geometry can be unit-tested on the JVM.
 *
 * @param previewAspect the preview surface's width / height.
 * @param left,top,right,bottom the brackets' bounds as fractions of the preview, 0..1.
 */
data class FramingRegion(
    val previewAspect: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {

    /** A crop in the decoded frame's own pixels. */
    data class Crop(val left: Int, val top: Int, val width: Int, val height: Int)

    /**
     * The brackets' bounds in the pixels of an upright [imageWidth] x [imageHeight] frame.
     *
     * Two steps, because the JPEG is not necessarily the field of view the user saw. `PreviewView`
     * is FILL_CENTER, so it shows the largest centred part of the frame with the preview's aspect
     * and hides the rest; an in-memory capture can still hand back the full sensor frame with its
     * viewport only as metadata. Finding the visible part first means this is correct either way —
     * when the capture was already cropped to the viewport, that step is a no-op.
     *
     * Returns the whole frame unchanged if the preview was never measured or the bounds are
     * degenerate, which is the honest fallback: better the old whole-frame behaviour than a crop
     * computed from a layout that does not exist yet.
     */
    fun cropIn(imageWidth: Int, imageHeight: Int): Crop {
        val whole = Crop(0, 0, imageWidth, imageHeight)
        if (imageWidth <= 0 || imageHeight <= 0) return whole
        if (!previewAspect.isFinite() || previewAspect <= 0f) return whole
        if (right <= left || bottom <= top) return whole

        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val visibleWidth: Float
        val visibleHeight: Float
        if (imageAspect > previewAspect) {
            // Frame is wider than the preview: its sides were never on screen.
            visibleHeight = imageHeight.toFloat()
            visibleWidth = visibleHeight * previewAspect
        } else {
            visibleWidth = imageWidth.toFloat()
            visibleHeight = visibleWidth / previewAspect
        }
        val offsetX = (imageWidth - visibleWidth) / 2f
        val offsetY = (imageHeight - visibleHeight) / 2f

        val l = (offsetX + left * visibleWidth).roundToInt().coerceIn(0, imageWidth - 1)
        val t = (offsetY + top * visibleHeight).roundToInt().coerceIn(0, imageHeight - 1)
        val r = (offsetX + right * visibleWidth).roundToInt().coerceIn(l + 1, imageWidth)
        val b = (offsetY + bottom * visibleHeight).roundToInt().coerceIn(t + 1, imageHeight)
        return Crop(l, t, r - l, b - t)
    }

    companion object {
        /**
         * The brackets' bounds relative to the preview they were drawn over, both in the same
         * coordinate space. Fractions are clamped, so brackets hanging off the edge of a short
         * preview crop to its edge rather than off the frame.
         */
        fun of(
            previewLeft: Float,
            previewTop: Float,
            previewWidth: Float,
            previewHeight: Float,
            guideLeft: Float,
            guideTop: Float,
            guideRight: Float,
            guideBottom: Float,
        ): FramingRegion? {
            if (previewWidth <= 0f || previewHeight <= 0f) return null
            return FramingRegion(
                previewAspect = previewWidth / previewHeight,
                left = ((guideLeft - previewLeft) / previewWidth).coerceIn(0f, 1f),
                top = ((guideTop - previewTop) / previewHeight).coerceIn(0f, 1f),
                right = ((guideRight - previewLeft) / previewWidth).coerceIn(0f, 1f),
                bottom = ((guideBottom - previewTop) / previewHeight).coerceIn(0f, 1f),
            )
        }
    }
}
