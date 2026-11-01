package my.edu.um.palmripeness.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The framing crop is pure geometry, so it is checked here rather than on a device.
 *
 * What these guard is the mapping the accuracy argument rests on: if the crop drifts off the
 * brackets, the model is fed something other than what the user framed, and the 45.7-point
 * finding in `results/framing_shift/` applies again without anything visibly breaking.
 */
class FramingRegionTest {

    /** The brackets as the handoff draws them: 280 x 360 dp on a 360 x 800 dp preview. */
    private fun reference(): FramingRegion = FramingRegion.of(
        previewLeft = 0f,
        previewTop = 0f,
        previewWidth = 360f,
        previewHeight = 800f,
        guideLeft = 40f,
        guideTop = 200f,
        guideRight = 320f,
        guideBottom = 560f,
    )!!

    @Test
    fun `brackets map onto a frame of the same aspect`() {
        // 1080 x 2400 is the reference preview at 3x, so the crop is the bracket dp x 3.
        val crop = reference().cropIn(1080, 2400)

        assertEquals(120, crop.left)
        assertEquals(600, crop.top)
        assertEquals(840, crop.width)
        assertEquals(1080, crop.height)
    }

    @Test
    fun `a wider frame than the preview is cut to what was on screen first`() {
        // A 4:3 sensor frame behind a 9:20 preview: FILL_CENTER hid the sides, so the brackets
        // are placed inside the visible slice, not inside the whole frame.
        val crop = reference().cropIn(3200, 2400)

        val visibleWidth = 2400f * (360f / 800f)
        val offsetX = (3200 - visibleWidth) / 2f
        assertEquals((offsetX + 40f / 360f * visibleWidth).toInt(), crop.left)
        assertEquals(600, crop.top)
        assertEquals((280f / 360f * visibleWidth).toInt(), crop.width)
        assertEquals(1080, crop.height)
    }

    @Test
    fun `the crop keeps the brackets' aspect ratio`() {
        val crop = reference().cropIn(1080, 2400)

        assertEquals(280f / 360f, crop.width.toFloat() / crop.height.toFloat(), 0.01f)
    }

    @Test
    fun `an unmeasured preview yields no region rather than a wrong one`() {
        assertNull(
            FramingRegion.of(0f, 0f, 0f, 0f, 40f, 200f, 320f, 560f)
        )
    }

    @Test
    fun `a degenerate region falls back to the whole frame`() {
        val crop = FramingRegion(previewAspect = 0.45f, left = 0.5f, top = 0.5f, right = 0.5f, bottom = 0.5f)
            .cropIn(1080, 2400)

        assertEquals(0, crop.left)
        assertEquals(0, crop.top)
        assertEquals(1080, crop.width)
        assertEquals(2400, crop.height)
    }

    @Test
    fun `brackets overhanging the preview clamp to it`() {
        val region = FramingRegion.of(
            previewLeft = 0f,
            previewTop = 0f,
            previewWidth = 360f,
            previewHeight = 400f,
            guideLeft = -20f,
            guideTop = 200f,
            guideRight = 380f,
            guideBottom = 560f,
        )!!
        val crop = region.cropIn(360, 400)

        assertEquals(0, crop.left)
        assertEquals(360, crop.width)
        assertEquals(400, crop.top + crop.height)
    }
}
