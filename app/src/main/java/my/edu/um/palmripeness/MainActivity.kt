package my.edu.um.palmripeness

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import my.edu.um.palmripeness.ui.components.SystemBarIcons
import my.edu.um.palmripeness.ui.screens.CameraScreen
import my.edu.um.palmripeness.ui.screens.HomeScreen
import my.edu.um.palmripeness.ui.screens.ResultsScreen
import my.edu.um.palmripeness.ui.screens.ReviewScreen
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmRipenessTheme

/** The app is one activity; the four screens are Compose destinations inside it. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PalmRipenessTheme {
                PalmRipenessApp()
            }
        }
    }
}

/** Route names, matching the flow diagram in `app/DESIGN_HANDOFF.md`. */
object Routes {
    const val HOME = "home"
    const val CAMERA = "camera"
    const val REVIEW = "review"
    const val RESULTS = "results"
}

@Composable
fun PalmRipenessApp(viewModel: MainViewModel = viewModel()) {
    val navController = rememberNavController()
    val modelState by viewModel.modelState.collectAsStateWithLifecycle()
    val stageLabels by viewModel.stageLabels.collectAsStateWithLifecycle()
    val photo by viewModel.pendingPhoto.collectAsStateWithLifecycle()
    val analysis by viewModel.analysis.collectAsStateWithLifecycle()

    // The system photo picker needs no storage permission and shows only what the user chooses,
    // which is the only way to stay consistent with an app that asks for nothing but the camera.
    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        viewModel.acceptPickedImage(uri) { decoded ->
            navController.goTo(if (decoded) Routes.REVIEW else Routes.RESULTS)
        }
    }
    val launchPicker = {
        pickImage.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
    }

    NavHost(
        navController = navController,
        startDestination = Routes.HOME,
        modifier = Modifier.fillMaxSize().background(Palm.Ground),
    ) {
        composable(Routes.HOME) {
            SystemBarIcons(darkIcons = true)
            HomeScreen(
                modelState = modelState,
                stageLabels = stageLabels,
                onTakePhoto = { navController.goTo(Routes.CAMERA) },
                onUploadImage = { launchPicker() },
                modifier = Modifier.windowInsetsPadding(WindowInsets.systemBars),
            )
        }

        composable(Routes.CAMERA) {
            SystemBarIcons(darkIcons = false)
            CameraScreen(
                onCaptured = { jpeg, region ->
                    viewModel.acceptCapture(jpeg, region) { decoded ->
                        navController.goTo(if (decoded) Routes.REVIEW else Routes.RESULTS)
                    }
                },
                onCaptureFailed = { navController.goTo(Routes.RESULTS) },
                onPickImage = { launchPicker() },
                onClose = { navController.backToHome() },
            )
        }

        composable(Routes.REVIEW) {
            SystemBarIcons(darkIcons = false)
            ReviewScreen(
                photo = photo,
                onRetake = { navController.goTo(Routes.CAMERA) },
                onDetect = {
                    viewModel.analyse()
                    navController.goTo(Routes.RESULTS)
                },
            )
        }

        composable(Routes.RESULTS) {
            SystemBarIcons(darkIcons = true)
            ResultsScreen(
                analysis = analysis,
                photo = photo,
                onNewPhoto = {
                    viewModel.clearPhotoAndAnalysis()
                    navController.goTo(Routes.CAMERA)
                },
                onDone = {
                    viewModel.clearPhotoAndAnalysis()
                    navController.backToHome()
                },
                onCancel = { navController.popBackStack() },
                onChooseAnother = { launchPicker() },
                modifier = Modifier.windowInsetsPadding(WindowInsets.systemBars),
            )
        }
    }
}

/**
 * Navigating forward without stacking duplicates. Retake → Camera → Review → Retake would
 * otherwise grow the back stack indefinitely, and the flow in the handoff is a small cycle, not
 * a history to walk back through.
 */
private fun NavHostController.goTo(route: String) {
    navigate(route) { launchSingleTop = true }
}

private fun NavHostController.backToHome() {
    navigate(Routes.HOME) {
        popUpTo(Routes.HOME) { inclusive = true }
        launchSingleTop = true
    }
}
