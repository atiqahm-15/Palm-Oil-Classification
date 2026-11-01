package my.edu.um.palmripeness.model

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The ≤ 2 s on-device latency requirement (NFR-2), measured on the phone rather than inferred.
 *
 * Two things are settled here that nothing else in the project can settle:
 *
 *  1. **Does the shipped model meet NFR-2 on real hardware.** Every latency figure quoted so far
 *     — 47.2 ms for FastViT_T8 float16 — is single-threaded x86 dev CPU from the conversion spike.
 *     `docs/report_limitations.md` L4 explicitly declines to claim compliance on that basis.
 *
 *  2. **Whether float16 is still the right deployment artefact.** The spike chose it over
 *     dynamic_int8 on a 47.2 ms vs 114.3 ms gap. That gap was measured on x86, where int8 has no
 *     specialised path; on ARM, int8 goes through XNNPACK and commonly wins. If the ordering
 *     reverses here, the bundled artefact should change — so this sweeps every staged mode rather
 *     than timing only the one already in `assets/`.
 *
 * Thread counts are swept too, because [RipenessClassifier] defaults to 4 while every published
 * figure in this project is single-threaded. Both are reported: 1 thread is the comparable number,
 * 4 is the one the user actually experiences.
 *
 * The report is written to the app's external files dir and echoed to stdout. Pull it with:
 *
 *     adb pull /sdcard/Android/data/my.edu.um.palmripeness/files/device_benchmark.md results/
 *
 * Assets come from `src/training/make_app_bundle.py` and are gitignored — regenerate them before
 * running this, or the sweep finds no modes and the test fails saying so.
 */
@RunWith(AndroidJUnit4::class)
class DeviceLatencyBenchmark {

    private val testAssets get() = InstrumentationRegistry.getInstrumentation().context.assets
    private val appContext get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** Discarded runs that pay for lazy kernel setup, delegate init and the first cache fill. */
    private val warmupRuns = 15

    /** Timed runs per (mode, threads) cell. 50 keeps a full sweep near a minute on a mid-range phone. */
    private val timedRuns = 50

    @Test
    fun sweepQuantizationModesAndThreadCounts() {
        val manifest = JSONObject(
            testAssets.open("benchmark/manifest.json").bufferedReader().use { it.readText() }
        )
        val modes = manifest.getJSONArray("modes")
        assertTrue(
            "No benchmark modes staged. Run `python src/training/make_app_bundle.py` first — " +
                "the .tflite variants are gitignored.",
            modes.length() > 0
        )

        val tensor = firstFixtureTensor()
        val bundledMode = manifest.getString("bundled_mode")
        val nfrMillis = manifest.optLong("nfr_latency_ms", 2000L)

        val rows = mutableListOf<Row>()
        for (i in 0 until modes.length()) {
            val entry = modes.getJSONObject(i)
            val mode = entry.getString("mode")
            for (threads in intArrayOf(1, 4)) {
                rows += measure(entry.getString("asset"), mode, threads, tensor, entry)
            }
        }

        val report = renderReport(rows, bundledMode, nfrMillis, manifest)
        println(report)
        // One log line per report line, under one tag, so the whole table can be recovered from
        // logcat alone — which is all a cloud device returns unless directories are pulled too.
        report.lineSequence().forEach { Log.i(LOG_TAG, it) }
        val dest = File(appContext.getExternalFilesDir(null), "device_benchmark.md")
        dest.writeText(report)
        println("Wrote ${dest.absolutePath}")

        // The gate itself. Checked on the mode the app actually ships, at the thread count it
        // actually uses — a different cell passing would not evidence the requirement.
        val shipped = rows.firstOrNull { it.mode == bundledMode && it.threads == 4 }
        assertTrue(
            "the bundled mode '$bundledMode' was not among the staged variants, so NFR-2 was not " +
                "gated. Re-run make_app_bundle.py with $bundledMode in --benchmark-modes.",
            shipped != null
        )
        assertTrue(
            "median inference ${shipped!!.medianMs} ms for $bundledMode exceeds the $nfrMillis ms requirement",
            shipped.medianMs < nfrMillis
        )
    }

    private fun measure(
        asset: String,
        mode: String,
        threads: Int,
        tensor: FloatArray,
        entry: JSONObject,
    ): Row {
        val options = Interpreter.Options().setNumThreads(threads)
        Interpreter(loadAsset(asset), options).use { interpreter ->
            interpreter.allocateTensors()
            val outputSize = interpreter.getOutputTensor(0).shape()[1]
            val input = ByteBuffer
                .allocateDirect(Preprocess.SIZE * Preprocess.SIZE * 3 * 4)
                .order(ByteOrder.nativeOrder())
            input.asFloatBuffer().put(tensor)
            val output = Array(1) { FloatArray(outputSize) }

            repeat(warmupRuns) {
                input.rewind()
                interpreter.run(input, output)
            }

            val samples = LongArray(timedRuns)
            for (i in 0 until timedRuns) {
                input.rewind()
                val started = System.nanoTime()
                interpreter.run(input, output)
                samples[i] = System.nanoTime() - started
            }
            samples.sort()

            return Row(
                mode = mode,
                threads = threads,
                sizeBytes = entry.getLong("size_bytes"),
                hostMs = if (entry.has("host_latency_ms_devcpu")) {
                    entry.getDouble("host_latency_ms_devcpu")
                } else {
                    null
                },
                medianMs = samples[samples.size / 2] / 1_000_000.0,
                minMs = samples.first() / 1_000_000.0,
                p95Ms = samples[(samples.size * 95) / 100] / 1_000_000.0,
                maxMs = samples.last() / 1_000_000.0,
            )
        }
    }

    /**
     * Times a real input, not zeros. A zero tensor is not representative — it can take different
     * paths through denormal handling and through whatever the delegate decides to skip.
     */
    private fun firstFixtureTensor(): FloatArray {
        val reference = JSONObject(
            testAssets.open("fixtures/reference.json").bufferedReader().use { it.readText() }
        )
        val id = reference.getJSONArray("images").getJSONObject(0).getString("id")
        val bytes = testAssets.open("fixtures/$id.input.bin").use { it.readBytes() }
        val floats = FloatArray(Preprocess.SIZE * Preprocess.SIZE * 3)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        return floats
    }

    /**
     * Reads the model into a direct buffer rather than memory-mapping it.
     *
     * The app memory-maps from its own APK (see [RipenessClassifier.mapAsset]), which needs the
     * asset left uncompressed. These variants live in the *test* APK, so rather than depend on
     * that packaging rule holding there too, they are read into memory — a one-off cost outside
     * the timed section, and it cannot silently change what is being measured.
     */
    private fun loadAsset(asset: String): ByteBuffer {
        val bytes = testAssets.open(asset).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
    }

    companion object {
        /** Filter logcat by this tag to get the benchmark report and the parity measurements. */
        const val LOG_TAG = "PalmBench"
    }

    private data class Row(
        val mode: String,
        val threads: Int,
        val sizeBytes: Long,
        val hostMs: Double?,
        val medianMs: Double,
        val minMs: Double,
        val p95Ms: Double,
        val maxMs: Double,
    ) {
        val sizeMb: Double get() = sizeBytes / 1048576.0
    }

    private fun renderReport(
        rows: List<Row>,
        bundledMode: String,
        nfrMillis: Long,
        manifest: JSONObject,
    ): String = buildString {
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
        appendLine("# On-device latency benchmark — Phase 3 (NFR-2)")
        appendLine()
        appendLine("Generated by `DeviceLatencyBenchmark` (`./gradlew :app:connectedDebugAndroidTest`).")
        appendLine("This replaces the conversion spike's x86 latency column, which was never a")
        appendLine("measurement of the target hardware.")
        appendLine()
        appendLine("| Property | Value |")
        appendLine("|---|---|")
        appendLine("| Device | ${Build.MANUFACTURER} ${Build.MODEL} (`${Build.DEVICE}`) |")
        appendLine("| SoC | $soc |")
        appendLine("| ABI | ${Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"} |")
        appendLine("| Android | ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) |")
        appendLine("| Source run | ${manifest.getString("source_run")} |")
        appendLine("| Bundled mode | $bundledMode |")
        appendLine("| Warm-up / timed runs | $warmupRuns / $timedRuns per cell |")
        appendLine("| Requirement | $nfrMillis ms |")
        appendLine()
        appendLine("| Mode | Threads | Size (MB) | Median (ms) | Min | p95 | Max | Host x86 (ms) | Within ${nfrMillis}ms |")
        appendLine("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for (r in rows) {
            val host = r.hostMs?.let { String.format("%.1f", it) } ?: "—"
            appendLine(
                "| ${r.mode}${if (r.mode == bundledMode) " **(shipped)**" else ""} | ${r.threads} | " +
                    String.format("%.1f", r.sizeMb) + " | " +
                    String.format("%.1f", r.medianMs) + " | " +
                    String.format("%.1f", r.minMs) + " | " +
                    String.format("%.1f", r.p95Ms) + " | " +
                    String.format("%.1f", r.maxMs) + " | $host | " +
                    (if (r.medianMs < nfrMillis) "yes" else "**NO**") + " |"
            )
        }
        appendLine()
        appendLine("## Does the host ordering survive?")
        appendLine()
        val atFour = rows.filter { it.threads == 4 }.sortedBy { it.medianMs }
        if (atFour.isNotEmpty()) {
            appendLine("Fastest to slowest on this device at 4 threads: " +
                atFour.joinToString(", ") { "${it.mode} (${String.format("%.1f", it.medianMs)} ms)" } + ".")
            val fastest = atFour.first()
            if (fastest.mode != bundledMode) {
                appendLine()
                appendLine("**${fastest.mode} is faster here than the shipped $bundledMode.** Before acting on")
                appendLine("that, check its accuracy in `results/tflite_export_summary.md` — the fastest mode")
                appendLine("is not automatically the right one, and full_int8 is excluded from this sweep for")
                appendLine("exactly that reason. A latency win only counts if fidelity holds.")
            } else {
                appendLine()
                appendLine("The shipped mode is also the fastest on this device, so the artefact choice made")
                appendLine("on host evidence holds on ARM.")
            }
        }
        appendLine()
        appendLine("## Reading these numbers")
        appendLine()
        appendLine("- One device, one thermal state. The sweep runs back to back, so later cells are")
        appendLine("  measured on a warmer SoC than earlier ones; a phone that throttles will show it as")
        appendLine("  a rising max relative to median. Re-run from cold to check anything borderline.")
        appendLine("- Interpreter time only — decode, EXIF and resize are not included. The user-visible")
        appendLine("  figure is larger, and the app's Analysing screen holds a 400 ms floor regardless.")
        appendLine("- Single-image inference, batch of 1, which is what the app does.")
    }
}
