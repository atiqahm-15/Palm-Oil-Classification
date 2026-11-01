package my.edu.um.palmripeness.ui.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import my.edu.um.palmripeness.R
import my.edu.um.palmripeness.camera.LumaMeter
import my.edu.um.palmripeness.model.FramingRegion
import my.edu.um.palmripeness.ui.components.CameraRoundButton
import my.edu.um.palmripeness.ui.components.FlashToggle
import my.edu.um.palmripeness.ui.components.FramingGuide
import my.edu.um.palmripeness.ui.components.GalleryButton
import my.edu.um.palmripeness.ui.components.HintPill
import my.edu.um.palmripeness.ui.components.Shutter
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import java.util.concurrent.Executors
import kotlin.coroutines.suspendCoroutine

private const val TAG = "PalmRipeness"

/** How long to wait for the preview to be measured before binding without a viewport. */
private const val MAX_LAYOUT_FRAMES = 60

/**
 * Camera. Full-bleed preview with a framing guide, a torch toggle and a shutter.
 *
 * The capture is bound to the preview viewport, so the JPEG contains exactly what the user saw.
 * Without that, a full-bleed preview on a tall screen shows a slice of the sensor field while the
 * saved photo would contain the rest of it too, and the user would be framing one image while the
 * model was fed another.
 *
 * The brackets are not advisory: what they hold is measured against the preview and handed up with
 * the JPEG as a [FramingRegion], and that is the part of the frame the model is given. Feeding it
 * the whole frame instead costs 45.7 points under simulation — see `results/framing_shift/`.
 */
@Composable
fun CameraScreen(
    onCaptured: (ByteArray, FramingRegion?) -> Unit,
    onCaptureFailed: () -> Unit,
    onPickImage: () -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(context.hasCameraPermission()) }
    var asked by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { result ->
        granted = result
        asked = true
    }

    LaunchedEffect(Unit) {
        if (!granted) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    if (!granted) {
        CameraAccessOffScreen(
            // Until the dialog has been answered there is nothing to say yet, so the screen stays
            // dark rather than accusing the user of denying something they were never asked.
            showReason = asked,
            onOpenSettings = { context.openAppSettings() },
            onPickImage = onPickImage,
            onClose = onClose,
            modifier = modifier,
        )
        return
    }

    LivePreview(
        onCaptured = onCaptured,
        onCaptureFailed = onCaptureFailed,
        onPickImage = onPickImage,
        onClose = onClose,
        modifier = modifier,
    )
}

@Composable
private fun LivePreview(
    onCaptured: (ByteArray, FramingRegion?) -> Unit,
    onCaptureFailed: () -> Unit,
    onPickImage: () -> Unit,
    onClose: () -> Unit,
    modifier: Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { analysisExecutor.shutdown() } }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var cameraControl by remember { mutableStateOf<CameraControl?>(null) }
    var torchAvailable by remember { mutableStateOf(false) }
    var torchOn by remember { mutableStateOf(false) }
    var tooDark by remember { mutableStateOf(false) }
    var capturing by remember { mutableStateOf(false) }

    // Set when the camera cannot be brought up at all. Without it the screen stays black with a
    // shutter that can never fire, and nothing tells the user why or what to do instead.
    var cameraFailed by remember { mutableStateOf(false) }

    // Both measured in the same coordinate space, so the brackets can be expressed as a fraction
    // of the preview and applied to a captured frame of any resolution.
    var previewBounds by remember { mutableStateOf<Rect?>(null) }
    var guideBounds by remember { mutableStateOf<Rect?>(null) }
    val framingRegion = remember(previewBounds, guideBounds) {
        val preview = previewBounds
        val guide = guideBounds
        if (preview == null || guide == null) {
            null
        } else {
            FramingRegion.of(
                previewLeft = preview.left,
                previewTop = preview.top,
                previewWidth = preview.width,
                previewHeight = preview.height,
                guideLeft = guide.left,
                guideTop = guide.top,
                guideRight = guide.right,
                guideBottom = guide.bottom,
            )
        }
    }

    LaunchedEffect(previewView) {
        // The viewport only exists once the view has been measured, and it is what ties the
        // captured frame to the previewed one. If it never arrives, bind without it rather than
        // leaving the user looking at a dead camera.
        var frames = 0
        while (previewView.viewPort == null && frames < MAX_LAYOUT_FRAMES) {
            withFrameNanos { }
            frames++
        }

        val provider = runCatching { context.awaitCameraProvider() }
            .onFailure { Log.e(TAG, "camera provider unavailable", it) }
            .getOrNull()
        if (provider == null) {
            cameraFailed = true
            return@LaunchedEffect
        }

        val preview = Preview.Builder().build()
            .also { it.surfaceProvider = previewView.surfaceProvider }
        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        val luma = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(analysisExecutor, LumaMeter { dark -> tooDark = dark }) }

        val group = UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(capture)
            .addUseCase(luma)
            .apply { previewView.viewPort?.let { setViewPort(it) } }
            .build()

        runCatching {
            provider.unbindAll()
            val camera = provider.bindToLifecycle(
                lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group
            )
            torchAvailable = camera.cameraInfo.hasFlashUnit()
            cameraControl = camera.cameraControl
            imageCapture = capture
        }.onFailure {
            Log.e(TAG, "could not bind the camera", it)
            cameraFailed = true
        }
    }

    if (cameraFailed) {
        CameraUnavailableScreen(onPickImage = onPickImage, onClose = onClose, modifier = modifier)
        return
    }

    // The toggle is the single source of truth for the torch, pushed to the camera whenever either
    // changes, so a tap that lands before binding completes is not silently lost.
    LaunchedEffect(torchOn, cameraControl) {
        runCatching { cameraControl?.enableTorch(torchOn) }
            .onFailure { Log.w(TAG, "torch unavailable", it) }
    }

    Box(modifier = modifier.fillMaxSize().background(Palm.Cam)) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxSize()
                .onGloballyPositioned { previewBounds = it.boundsInWindow() },
        )

        // The guide and the hint are one block: the handoff puts the pill's top edge at 530dp
        // and the guide's bottom at 510dp on its 360x800 reference, so the real relationship is
        // "20dp under the brackets" and that is what survives a taller screen.
        //
        // The block is centred in the space above the controls rather than pinned 150dp below
        // the status bar. A fixed offset reproduces the reference exactly and collides on
        // anything shorter: status bar + 150 + 360 (guide) + 20 + pill leaves about 8dp of
        // clearance at 360x800, and overlaps the shutter row outright on a 640dp-tall screen,
        // which minSdk 26 still admits. Centring keeps the reference layout within a few dp and
        // degrades by shrinking the margins instead of stacking controls on top of each other.
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(bottom = Dim.cameraGuideReserve)
                .fillMaxSize()
                .wrapContentHeight(Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            FramingGuide(
                modifier = Modifier.onGloballyPositioned { guideBounds = it.boundsInWindow() }
            )
            Spacer(Modifier.height(20.dp))
            HintPill(
                text = if (tooDark) {
                    "Too dark. Turn on the flash or move into light."
                } else {
                    // The brackets are what gets classified, so the hint has to point at them
                    // rather than at the frame.
                    "Fill the brackets with one bunch"
                }
            )
        }

        if (torchAvailable) {
            FlashToggle(
                on = torchOn,
                onIconRes = R.drawable.ic_flash_on,
                offIconRes = R.drawable.ic_flash_off,
                onToggle = { torchOn = !torchOn },
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .windowInsetsPadding(WindowInsets.statusBars)
                    .padding(top = 52.dp, end = 20.dp),
            )
        }

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(
                    start = Dim.screenSide,
                    end = Dim.screenSide,
                    bottom = Dim.cameraControlsBottom,
                ),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            GalleryButton(iconRes = R.drawable.ic_gallery, onClick = onPickImage)
            Shutter(
                enabled = imageCapture != null && !capturing,
                onClick = {
                    val capture = imageCapture ?: return@Shutter
                    capturing = true
                    capture.takePicture(
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageCapturedCallback() {
                            override fun onCaptureSuccess(image: ImageProxy) {
                                val jpeg = image.toJpegBytes()
                                image.close()
                                capturing = false
                                onCaptured(jpeg, framingRegion)
                            }

                            override fun onError(exception: ImageCaptureException) {
                                Log.e(TAG, "capture failed", exception)
                                capturing = false
                                onCaptureFailed()
                            }
                        },
                    )
                },
            )
            CameraRoundButton(
                iconRes = R.drawable.ic_close,
                contentDescription = "Close camera",
                onClick = onClose,
            )
        }
    }
}

private fun Context.hasCameraPermission(): Boolean =
    ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

private fun Context.openAppSettings() {
    startActivity(
        Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", packageName, null),
        )
    )
}

/**
 * `future.get()` throws when the provider cannot be created, and it has to be called inside the
 * listener. Resuming with the exception rather than letting it fly is what makes the caller's
 * `runCatching` able to see it: thrown from the listener it would land on the main executor
 * instead, crash the process, and leave this continuation never resumed.
 */
private suspend fun Context.awaitCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            { continuation.resumeWith(runCatching { future.get() }) },
            ContextCompat.getMainExecutor(this),
        )
    }

/** In-memory capture hands back a single-plane JPEG; these are those bytes. */
private fun ImageProxy.toJpegBytes(): ByteArray {
    val buffer = planes[0].buffer
    buffer.rewind()
    return ByteArray(buffer.remaining()).also { buffer.get(it) }
}
