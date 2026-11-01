package my.edu.um.palmripeness.model

import android.content.Context
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * The TFLite interpreter wrapper. Loads the bundled FastViT_T8 float16 export, its labels, and
 * its contract, and runs one image at a time.
 *
 * Nothing here reaches the network: the model is memory-mapped out of the APK.
 */
class RipenessClassifier private constructor(
    private val interpreter: Interpreter,
    val labels: List<String>,
    val contract: ModelContract,
) : Closeable {

    private val lock = Any()
    private val input = ByteBuffer.allocateDirect(Preprocess.SIZE * Preprocess.SIZE * 3 * 4)
        .order(java.nio.ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(labels.size) }

    /**
     * Runs the model on a 224 x 224 x 3 tensor already scaled to [-1, 1].
     *
     * Call from a background thread: the handoff requires the screen never look frozen, and the
     * interpreter blocks.
     */
    fun classify(tensor: FloatArray): Prediction = synchronized(lock) {
        require(tensor.size == Preprocess.SIZE * Preprocess.SIZE * 3) {
            "expected ${Preprocess.SIZE * Preprocess.SIZE * 3} floats, got ${tensor.size}"
        }
        input.rewind()
        input.asFloatBuffer().put(tensor)
        input.rewind()

        val started = SystemClock.elapsedRealtime()
        interpreter.run(input, output)
        val elapsed = SystemClock.elapsedRealtime() - started

        Prediction(labels, output[0].copyOf(), elapsed)
    }

    /**
     * Takes the same lock [classify] holds, because freeing the interpreter under a running
     * inference is a native-heap free of memory the interpreter is still reading: it crashes the
     * process rather than throwing. Cancelling `viewModelScope` does not interrupt a blocking
     * `interpreter.run`, so `onCleared()` can genuinely arrive mid-inference, and the lock makes
     * close wait for the run to finish instead.
     */
    override fun close() = synchronized(lock) { interpreter.close() }

    companion object {
        /**
         * Four threads by default. The parity fixtures were produced with a single thread in
         * Python, so a test that needs to reproduce them exactly should pass `numThreads = 1`.
         */
        fun load(context: Context, numThreads: Int = 4): RipenessClassifier {
            val contract = ModelContract.load(context)
            val labels = context.assets.open(contract.labelsFile).bufferedReader().use { reader ->
                reader.readLines().map { it.trim() }.filter { it.isNotEmpty() }
            }
            contract.verify(labels)

            val options = Interpreter.Options().setNumThreads(numThreads)
            val interpreter = Interpreter(mapAsset(context, contract.modelFile), options)
            interpreter.allocateTensors()

            val shape = interpreter.getInputTensor(0).shape().toList()
            check(shape == contract.inputShape) {
                "model input $shape disagrees with the contract's ${contract.inputShape}"
            }
            val outShape = interpreter.getOutputTensor(0).shape().toList()
            check(outShape == listOf(1, labels.size)) {
                "model output $outShape but ${labels.size} labels"
            }
            return RipenessClassifier(interpreter, labels, contract)
        }

        /** Memory-maps the asset; `noCompress += "tflite"` in the build script is what allows it. */
        private fun mapAsset(context: Context, name: String): ByteBuffer =
            context.assets.openFd(name).use { fd ->
                fd.createInputStream().use { stream ->
                    stream.channel.map(
                        FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                    )
                }
            }
    }
}
