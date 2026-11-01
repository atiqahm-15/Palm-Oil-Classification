package my.edu.um.palmripeness

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import my.edu.um.palmripeness.model.FramingRegion
import my.edu.um.palmripeness.model.ImageInput
import my.edu.um.palmripeness.model.ModelContract
import my.edu.um.palmripeness.model.Prediction
import my.edu.um.palmripeness.model.RipenessClassifier

/**
 * Holds what outlives a screen: the loaded interpreter, the photo waiting to be analysed, and the
 * analysis itself.
 *
 * Model loading happens off the main thread at start-up, and a failure becomes the Home screen's
 * "model did not load" state rather than a crash — a thesis demo that dies on launch tells nobody
 * what went wrong.
 *
 * The pending photo is deliberately *not* saved across process death. Restoring it would mean
 * writing a full-resolution bitmap somewhere, and the app has no storage permission and keeps
 * nothing; if the process is killed mid-flow the user retakes the photo.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    sealed interface ModelState {
        data object Loading : ModelState
        data class Ready(val classifier: RipenessClassifier) : ModelState
        data class Failed(val reason: String) : ModelState
    }

    /** Where the Results screen is in its own small lifecycle. */
    sealed interface Analysis {
        data object Idle : Analysis
        data object Running : Analysis
        data class Done(val prediction: Prediction) : Analysis

        /** The image would not decode, or the model could not be fed it. */
        data object Unreadable : Analysis
    }

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    /**
     * The stage names in output order, read from labels.txt rather than from an enum, so the
     * Home strip and the Results chart cannot disagree with the model. Available even when the
     * interpreter itself failed to load, which is why it is not read off [ModelState.Ready].
     */
    private val _stageLabels = MutableStateFlow(FALLBACK_LABELS)
    val stageLabels: StateFlow<List<String>> = _stageLabels.asStateFlow()

    private val _pendingPhoto = MutableStateFlow<Bitmap?>(null)

    /**
     * The photo, upright and — from the camera — already cropped to the framing brackets, exactly
     * as Review shows it and the model will see it.
     */
    val pendingPhoto: StateFlow<Bitmap?> = _pendingPhoto.asStateFlow()

    private val _analysis = MutableStateFlow<Analysis>(Analysis.Idle)
    val analysis: StateFlow<Analysis> = _analysis.asStateFlow()

    init {
        viewModelScope.launch {
            val context = getApplication<Application>()
            withContext(Dispatchers.IO) {
                runCatching {
                    val contract = ModelContract.load(context)
                    context.assets.open(contract.labelsFile).bufferedReader().use { reader ->
                        reader.readLines().map { it.trim() }.filter { it.isNotEmpty() }
                    }
                }.onSuccess { labels -> _stageLabels.value = labels }
                    .onFailure { Log.w(TAG, "labels.txt unreadable; using the built-in order", it) }

                _modelState.value = runCatching { RipenessClassifier.load(context) }
                    .fold(
                        onSuccess = { ModelState.Ready(it) },
                        onFailure = {
                            Log.e(TAG, "model failed to load", it)
                            ModelState.Failed(it.message ?: it::class.java.simpleName)
                        },
                    )
            }
        }
    }

    /**
     * Decodes a freshly captured JPEG and cuts it down to [region], the framing brackets the user
     * was aiming through. Cropping here rather than at inference keeps one photo in the app: what
     * Review shows is what the model is given.
     *
     * [onDecoded] receives false when the bytes will not decode, which the caller turns into the
     * unreadable-photo state on Results rather than a dead end on Review.
     */
    fun acceptCapture(jpeg: ByteArray, region: FramingRegion?, onDecoded: (Boolean) -> Unit) {
        viewModelScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                ImageInput.decodeUpright(jpeg)?.let { decoded ->
                    if (region == null) decoded else ImageInput.cropToFraming(decoded, region)
                }
            }
            finishDecode(bitmap, onDecoded)
        }
    }

    /** Same, for an image chosen through the system picker. */
    fun acceptPickedImage(uri: Uri, onDecoded: (Boolean) -> Unit) {
        viewModelScope.launch {
            val bitmap = withContext(Dispatchers.IO) {
                runCatching { ImageInput.decodeUpright(getApplication(), uri) }
                    .onFailure { Log.w(TAG, "could not read the chosen image", it) }
                    .getOrNull()
            }
            finishDecode(bitmap, onDecoded)
        }
    }

    private fun finishDecode(bitmap: Bitmap?, onDecoded: (Boolean) -> Unit) {
        setPending(bitmap)
        _analysis.value = if (bitmap == null) Analysis.Unreadable else Analysis.Idle
        onDecoded(bitmap != null)
    }

    /**
     * Runs the model on the pending photo.
     *
     * Held on screen for at least [MIN_ANALYSING_MILLIS] so the Analysing state reads as a step
     * rather than a flicker. Inference is tens of milliseconds, so this is a deliberate floor, not
     * a progress estimate — which is why the screen shows no percentage.
     */
    fun analyse() {
        val classifier = (_modelState.value as? ModelState.Ready)?.classifier
        val photo = _pendingPhoto.value
        if (classifier == null || photo == null) {
            _analysis.value = Analysis.Unreadable
            return
        }
        if (_analysis.value is Analysis.Running) return

        _analysis.value = Analysis.Running
        viewModelScope.launch {
            val started = SystemClock.elapsedRealtime()
            val result = withContext(Dispatchers.Default) {
                runCatching { classifier.classify(ImageInput.toModelInput(photo)) }
                    .onFailure { Log.e(TAG, "inference failed", it) }
                    .getOrNull()
            }
            delay((MIN_ANALYSING_MILLIS - (SystemClock.elapsedRealtime() - started)).coerceAtLeast(0))
            _analysis.value = result?.let { Analysis.Done(it) } ?: Analysis.Unreadable
        }
    }

    /** Leaving Results for a new photo: drop the old one so nothing stale can be re-shown. */
    fun clearPhotoAndAnalysis() {
        setPending(null)
        _analysis.value = Analysis.Idle
    }

    private fun setPending(bitmap: Bitmap?) {
        val previous = _pendingPhoto.value
        _pendingPhoto.value = bitmap
        if (previous != null && previous != bitmap) previous.recycle()
    }

    override fun onCleared() {
        setPending(null)
        (_modelState.value as? ModelState.Ready)?.classifier?.close()
        super.onCleared()
    }

    private companion object {
        const val TAG = "PalmRipeness"

        /** Decided 2026-09-24: a floor, not a progress bar. See app/DESIGN_NOTES.md. */
        const val MIN_ANALYSING_MILLIS = 400L

        /** Only reached if the asset itself is unreadable, in which case nothing will classify. */
        val FALLBACK_LABELS = listOf("unripe", "underripe", "ripe", "overripe")
    }
}
