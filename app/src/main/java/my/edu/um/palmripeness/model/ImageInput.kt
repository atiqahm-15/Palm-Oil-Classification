package my.edu.um.palmripeness.model

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Bridges Android imagery to [Preprocess]. Kept separate from it so the resize parity test can
 * run on the JVM without an emulator.
 */
object ImageInput {

    /**
     * Above this the source is sub-sampled on decode. Sub-sampling by a power of two before the
     * bilinear resize is a small departure from the single-step `tf.image.resize`, which is why
     * the jpg-path fixture test is checked with a tolerance rather than bit-exactly.
     *
     * The cap was 24M, which never fired for the photos it was supposed to protect against: a
     * 12 MP phone camera is 12M pixels, so it decoded at full resolution into 48 MB of
     * ARGB_8888, and [applyOrientation] then held a second copy for ~96 MB at once.
     *
     * 8M is the value that fixes that without disturbing the evidence. A 12 MP photo now decodes
     * at `inSampleSize = 2` to 3M pixels, so the peak is ~24 MB across both bitmaps instead of
     * ~96 MB, and a 48 MP one lands in the same place. The ceiling is set by the parity fixtures
     * rather than by taste: the largest is 7.68M pixels, so anything below 8M would start
     * sub-sampling them, and `ModelParityTest`'s jpg path — whose argmax assertion is the one
     * that must never bend — would be measuring a decode the tolerance was never derived from.
     */
    private const val MAX_SOURCE_PIXELS = 8_000_000

    /** Decodes [uri], applying EXIF orientation so the bunch is upright, as the contract requires. */
    fun decodeUpright(context: Context, uri: Uri): Bitmap? {
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readAllBytesCompat() }
            ?: return null
        return decodeUpright(bytes)
    }

    /** Decodes JPEG/PNG [bytes], applying EXIF orientation. */
    fun decodeUpright(bytes: ByteArray): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight)
        }
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: return null
        val orientation = runCatching {
            ExifInterface(bytes.inputStream()).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        return applyOrientation(decoded, orientation)
    }

    /**
     * Cuts [bitmap] down to what the camera's framing brackets held, recycling the original.
     *
     * Only the camera path has a region: a photo chosen from the gallery was framed by whoever
     * took it, and there is nothing to say where its subject sits, so it is fed whole.
     */
    fun cropToFraming(bitmap: Bitmap, region: FramingRegion): Bitmap {
        val crop = region.cropIn(bitmap.width, bitmap.height)
        if (crop.left == 0 && crop.top == 0 &&
            crop.width == bitmap.width && crop.height == bitmap.height
        ) {
            return bitmap
        }
        val cropped = Bitmap.createBitmap(bitmap, crop.left, crop.top, crop.width, crop.height)
        if (cropped != bitmap) bitmap.recycle()
        return cropped
    }

    private fun sampleSizeFor(width: Int, height: Int): Int {
        var sample = 1
        while (width.toLong() * height / (sample.toLong() * sample) > MAX_SOURCE_PIXELS) sample *= 2
        return sample
    }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
            else -> return bitmap
        }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }

    /**
     * Resize + rescale [bitmap] into the tensor the model expects: 224 x 224 x 3 float32, RGB,
     * values in [-1, 1].
     */
    fun toModelInput(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val argb = IntArray(width)
        val resized = Preprocess.resizeBilinear(bitmap.height, width) { y, out ->
            bitmap.getPixels(argb, 0, width, 0, y, width, 1)
            for (x in 0 until width) {
                val p = argb[x]
                val o = x * 3
                out[o] = ((p shr 16) and 0xFF).toFloat()
                out[o + 1] = ((p shr 8) and 0xFF).toFloat()
                out[o + 2] = (p and 0xFF).toFloat()
            }
        }
        return Preprocess.scaleToModelRange(resized)
    }

    /** Wraps [input] as the little-endian direct buffer the interpreter reads. */
    fun asInputBuffer(input: FloatArray): ByteBuffer =
        ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder()).apply {
            asFloatBuffer().put(input)
            rewind()
        }

    private fun InputStream.readAllBytesCompat(): ByteArray {
        val out = ByteArrayOutputStream()
        copyTo(out)
        return out.toByteArray()
    }
}
