package my.edu.um.palmripeness.model

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Does the app get the same numbers out of the model as Python did?
 *
 * Two separate checks, because there are two separate ways to be wrong:
 *
 *  - [interpreterMatchesPythonFromExactTensor] feeds the exact float32 tensor Python fed and
 *    demands the probabilities back to 1e-4. This tests the interpreter wiring only — buffer
 *    order, output shape, label order.
 *  - [preprocessingMatchesPythonFromJpeg] starts from the same JPEG and runs the app's own
 *    decode, resize and rescale. JPEG decoders differ slightly between TensorFlow and Android,
 *    so this is checked with a tolerance and on argmax agreement.
 *
 * Fixtures come from `src/training/make_app_bundle.py`. The `.jpg` and `.input.bin` files are
 * gitignored (dataset images stay out of git) — regenerate them locally before running this.
 */
@RunWith(AndroidJUnit4::class)
class ModelParityTest {

    private lateinit var classifier: RipenessClassifier
    private lateinit var reference: JSONObject

    private val fixtureAssets get() = InstrumentationRegistry.getInstrumentation().context.assets
    private val appContext get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun setUp() {
        // Single thread: the reference probabilities were produced with num_threads=1.
        classifier = RipenessClassifier.load(appContext, numThreads = 1)
        reference = JSONObject(
            fixtureAssets.open("fixtures/reference.json").bufferedReader().use { it.readText() }
        )
        val classNames = reference.getJSONArray("class_names")
        assertEquals(
            "fixture class order differs from labels.txt",
            (0 until classNames.length()).map { classNames.getString(it) },
            classifier.labels,
        )
    }

    @After
    fun tearDown() {
        classifier.close()
    }

    @Test
    fun interpreterMatchesPythonFromExactTensor() {
        val tolerance = reference.getDouble("tolerance_from_input_bin").toFloat()
        var worst = 0f
        var worstAt = ""
        forEachImage { id, expected, _ ->
            val tensor = readTensor("fixtures/$id.input.bin")
            val scores = classifier.classify(tensor).scores
            for (k in expected.indices) {
                val diff = abs(scores[k] - expected[k])
                if (diff > worst) {
                    worst = diff
                    worstAt = "$id class $k: expected ${expected[k]}, got ${scores[k]}"
                }
            }
        }
        report("parity/tensor: worst=$worst tolerance=$tolerance at $worstAt")
        assertTrue("worst probability difference $worst — $worstAt", worst <= tolerance)
    }

    @Test
    fun preprocessingMatchesPythonFromJpeg() {
        val tolerance = reference.getDouble("tolerance_from_jpg").toFloat()
        var worst = 0f
        var worstAt = ""
        var argmaxAgreements = 0
        var images = 0
        forEachImage { id, expected, expectedClass ->
            val bytes = fixtureAssets.open("fixtures/$id.jpg").use { it.readBytes() }
            val bitmap = requireNotNull(ImageInput.decodeUpright(bytes)) { "$id failed to decode" }
            val prediction = classifier.classify(ImageInput.toModelInput(bitmap))
            if (prediction.topLabel == expectedClass) argmaxAgreements++
            images++
            for (k in expected.indices) {
                val diff = abs(prediction.scores[k] - expected[k])
                if (diff > worst) {
                    worst = diff
                    worstAt = "$id class $k: expected ${expected[k]}, got ${prediction.scores[k]}"
                }
            }
        }
        report(
            "parity/jpg: argmax $argmaxAgreements/$images, worst=$worst tolerance=$tolerance " +
                "at $worstAt"
        )
        assertTrue("no fixtures found", images > 0)
        assertEquals(
            "the app's preprocessing changed the predicted class on some fixtures",
            images, argmaxAgreements,
        )
        assertTrue("worst probability difference $worst — $worstAt", worst <= tolerance)
    }

    @Test
    fun inferenceStaysWellUnderTheTwoSecondRequirement() {
        val tensor = readTensor("fixtures/${firstFixtureId()}.input.bin")
        classifier.classify(tensor) // warm up: the first call pays for lazy kernel setup
        val runs = (1..10).map { classifier.classify(tensor).inferenceMillis }
        val median = runs.sorted()[runs.size / 2]
        report(
            "parity/latency: median=${median}ms runs=${runs.sorted()} threads=1 " +
                "device=${Build.MANUFACTURER} ${Build.MODEL} api=${Build.VERSION.SDK_INT}"
        )
        assertTrue("median inference $median ms exceeds the 2000 ms requirement", median < 2000)
    }

    /**
     * A passing assertion records nothing, and on a cloud device (Firebase Test Lab) logcat is the
     * only output that comes back without extra setup — so the measured numbers are logged under
     * the same tag as [DeviceLatencyBenchmark]'s report, pass or fail.
     */
    private fun report(line: String) {
        Log.i(DeviceLatencyBenchmark.LOG_TAG, line)
    }

    private fun firstFixtureId(): String =
        reference.getJSONArray("images").getJSONObject(0).getString("id")

    private fun forEachImage(body: (id: String, expected: FloatArray, expectedClass: String) -> Unit) {
        val images = reference.getJSONArray("images")
        for (i in 0 until images.length()) {
            val image = images.getJSONObject(i)
            val probs = image.getJSONArray("probabilities")
            body(
                image.getString("id"),
                FloatArray(probs.length()) { probs.getDouble(it).toFloat() },
                image.getString("predicted_class"),
            )
        }
    }

    private fun readTensor(asset: String): FloatArray {
        val bytes = fixtureAssets.open(asset).use { it.readBytes() }
        val expected = Preprocess.SIZE * Preprocess.SIZE * 3
        assertEquals("$asset is not a ${expected}-float tensor", expected * 4, bytes.size)
        val floats = FloatArray(expected)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        return floats
    }
}
