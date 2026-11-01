package my.edu.um.palmripeness.model

import android.content.Context
import org.json.JSONObject

/**
 * `assets/model_contract.json`, written by `src/training/make_app_bundle.py` alongside the
 * model itself.
 *
 * The app reads the contract rather than trusting constants typed twice, and [verify] fails
 * loudly if a re-export changes an assumption this code is built on. A wrong assumption here
 * does not crash and does not look wrong — it just returns four confident, wrong scores.
 */
data class ModelContract(
    val modelFile: String,
    val labelsFile: String,
    val sourceRun: String,
    val sourceMode: String,
    val sha256: String,
    val sizeBytes: Long,
    val generated: String,
    val inputShape: List<Int>,
    val valueRange: List<Double>,
    val classNames: List<String>,
    val calibration: String,
) {
    /** Size for display, e.g. "6.1 MB". */
    val sizeMb: String get() = String.format("%.1f MB", sizeBytes / 1048576.0)

    /**
     * Checks the contract against what [Preprocess] and the UI actually implement. Called at
     * load time so a mismatch surfaces as the Home screen's "model did not load" state rather
     * than as silently wrong predictions.
     */
    fun verify(labels: List<String>) {
        check(inputShape == listOf(1, Preprocess.SIZE, Preprocess.SIZE, 3)) {
            "contract input shape $inputShape but Preprocess targets ${Preprocess.SIZE}"
        }
        check(valueRange == listOf(-1.0, 1.0)) {
            "contract value range $valueRange but Preprocess scales to [-1, 1]"
        }
        check(labels == classNames) {
            "labels.txt $labels disagrees with the contract's class_names $classNames"
        }
    }

    companion object {
        fun load(context: Context, asset: String = "model_contract.json"): ModelContract {
            val json = JSONObject(context.assets.open(asset).bufferedReader().use { it.readText() })
            val input = json.getJSONObject("input")
            val output = json.getJSONObject("output")
            return ModelContract(
                modelFile = json.getString("model_file"),
                labelsFile = json.getString("labels_file"),
                sourceRun = json.getString("source_run"),
                sourceMode = json.getString("source_mode"),
                sha256 = json.getString("sha256"),
                sizeBytes = json.getLong("size_bytes"),
                generated = json.getString("generated"),
                inputShape = input.getJSONArray("shape").toIntList(),
                valueRange = input.getJSONArray("value_range").toDoubleList(),
                classNames = json.getJSONArray("class_names").toStringList(),
                calibration = output.getString("calibration"),
            )
        }

        private fun org.json.JSONArray.toIntList() = (0 until length()).map { getInt(it) }
        private fun org.json.JSONArray.toDoubleList() = (0 until length()).map { getDouble(it) }
        private fun org.json.JSONArray.toStringList() = (0 until length()).map { getString(it) }
    }
}
