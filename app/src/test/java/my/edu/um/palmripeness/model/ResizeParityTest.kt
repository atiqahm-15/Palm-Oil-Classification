package my.edu.um.palmripeness.model

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * The Kotlin resize must reproduce `tf.image.resize` (bilinear, half-pixel centres, no
 * antialiasing), because that is what produced every pixel the model was trained and tested on.
 *
 * The fixture is `app/src/test/resources/resize_parity.json`, written by
 * `src/training/make_app_bundle.py`. It stores no image: the source is a formula, regenerated
 * here, and only sampled output values are recorded.
 */
class ResizeParityTest {

    /** Mirrors `pattern_pixel` in make_app_bundle.py. */
    private fun patternPixel(y: Int, x: Int, c: Int): Int =
        (x * 31 + y * 17 + c * 101 + (x * y) % 7 * 23) % 256

    private fun fixture(): JSONObject {
        val stream = requireNotNull(javaClass.getResourceAsStream("/resize_parity.json")) {
            "resize_parity.json missing from test resources — regenerate with make_app_bundle.py"
        }
        return JSONObject(stream.bufferedReader().use { it.readText() })
    }

    @Test
    fun `resize matches tensorflow at every sampled position`() {
        val fixture = fixture()
        assertEquals(
            "fixture was generated for a different input size",
            Preprocess.SIZE, fixture.getInt("target_size")
        )
        assertEquals(
            "the pattern formula changed; update patternPixel to match",
            "(x*31 + y*17 + c*101 + (x*y)%7*23) % 256", fixture.getString("pattern")
        )

        val cases: JSONArray = fixture.getJSONArray("cases")
        assertTrue("no cases in the fixture", cases.length() > 0)

        var worst = 0.0
        var worstAt = ""
        var checked = 0
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val srcHeight = case.getInt("src_height")
            val srcWidth = case.getInt("src_width")

            val resized = Preprocess.resizeBilinear(srcHeight, srcWidth) { y, out ->
                for (x in 0 until srcWidth) {
                    for (c in 0..2) out[x * 3 + c] = patternPixel(y, x, c).toFloat()
                }
            }

            val samples = case.getJSONArray("samples")
            for (s in 0 until samples.length()) {
                val sample = samples.getJSONArray(s)
                val y = sample.getInt(0)
                val x = sample.getInt(1)
                val c = sample.getInt(2)
                val expected = sample.getDouble(3)
                val actual = resized[(y * Preprocess.SIZE + x) * 3 + c].toDouble()
                val diff = abs(actual - expected)
                if (diff > worst) {
                    worst = diff
                    worstAt = "${srcWidth}x$srcHeight at ($y, $x, $c): expected $expected, got $actual"
                }
                checked++
            }
        }
        assertTrue("checked nothing", checked > 0)
        // 1e-3 on a 0..255 scale is well under a quantisation step, and any real mismatch
        // (wrong centre convention, aspect-preserving resize, integer arithmetic) is off by
        // whole units at least.
        assertTrue("worst difference $worst over $checked samples — $worstAt", worst < 1e-3)
    }

    @Test
    fun `scaling maps the pixel range onto the model range`() {
        val scaled = Preprocess.scaleToModelRange(floatArrayOf(0f, 127.5f, 255f))
        assertEquals(-1f, scaled[0], 1e-6f)
        assertEquals(0f, scaled[1], 1e-6f)
        assertEquals(1f, scaled[2], 1e-6f)
    }

    @Test
    fun `a uniform image resizes to the same uniform value`() {
        val resized = Preprocess.resizeBilinear(53, 97) { _, out -> out.fill(200f) }
        assertEquals(Preprocess.SIZE * Preprocess.SIZE * 3, resized.size)
        assertTrue(resized.all { abs(it - 200f) < 1e-4f })
    }
}
