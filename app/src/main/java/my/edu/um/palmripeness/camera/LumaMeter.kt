package my.edu.um.palmripeness.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * Estimates preview brightness so the camera screen can warn that a photo will be too dark.
 *
 * Averages a sparse sample of the Y plane — the luma channel is already there in YUV_420_888, so
 * this costs nothing and needs no colour conversion — and smooths it, because a single frame
 * flickers as autoexposure hunts and a hint that blinks is worse than no hint.
 *
 * **The thresholds are provisional.** `DESIGN_HANDOFF.md` lists "what brightness measure and
 * threshold trigger the low-light hint" as an open question, and these numbers were not derived
 * from field photographs: they are a starting point to calibrate on a real phone, where the same
 * scene should trip the hint exactly when a person would call it too dark to photograph. Entry and
 * exit differ so the hint does not chatter at the boundary.
 */
class LumaMeter(private val onDarkChanged: (Boolean) -> Unit) : ImageAnalysis.Analyzer {

    private var smoothed = -1f
    private var dark = false

    override fun analyze(image: ImageProxy) {
        try {
            val mean = meanLuma(image)
            smoothed = if (smoothed < 0f) mean else smoothed + SMOOTHING * (mean - smoothed)
            val nowDark = when {
                dark -> smoothed < DARK_EXIT
                else -> smoothed < DARK_ENTER
            }
            if (nowDark != dark) {
                dark = nowDark
                onDarkChanged(nowDark)
            }
        } finally {
            image.close()
        }
    }

    private fun meanLuma(image: ImageProxy): Float {
        val buffer = image.planes[0].buffer
        buffer.rewind()
        var total = 0L
        var count = 0
        var i = 0
        while (i < buffer.limit()) {
            total += buffer.get(i).toInt() and 0xFF
            count++
            i += SAMPLE_STRIDE
        }
        return if (count == 0) 0f else total.toFloat() / count
    }

    private companion object {
        /** Every 97th byte: coprime with typical row strides, so the sample is not one column. */
        const val SAMPLE_STRIDE = 97
        const val SMOOTHING = 0.2f
        const val DARK_ENTER = 50f
        const val DARK_EXIT = 70f
    }
}
