package my.edu.um.palmripeness.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.ui.components.FlexibleSpace
import my.edu.um.palmripeness.ui.components.OutlineButton
import my.edu.um.palmripeness.ui.components.ScreenColumn
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmRipenessTheme
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * Camera permission denied. Offers the two things that can still be done — change the setting, or
 * skip the camera entirely and pick an existing image — rather than only explaining the problem.
 *
 * The copy says where photos go, because the reason a permission prompt gets refused is usually
 * that nobody said.
 */
@Composable
fun CameraAccessOffScreen(
    showReason: Boolean,
    onOpenSettings: () -> Unit,
    onPickImage: () -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    ScreenColumn(modifier = modifier.background(Palm.Cam)) {
        if (showReason) {
            Text(
                text = "Camera access is off",
                style = PalmText.ScreenH1,
                color = Color.White,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = "Palm Ripeness needs the camera to photograph a bunch. Photos stay on " +
                    "this phone.",
                style = PalmText.Body,
                color = Color.White.copy(alpha = 0.86f),
            )
        }

        FlexibleSpace()

        if (showReason) {
            OutlineButton(
                text = "Open settings",
                onClick = onOpenSettings,
                onDark = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(12.dp))
            OutlineButton(
                text = "Upload an image instead",
                onClick = onPickImage,
                onDark = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(12.dp))
        }
        OutlineButton(
            text = "Close",
            onClick = onClose,
            onDark = true,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

/**
 * The camera was permitted but could not be brought up: no provider, or `bindToLifecycle` failed.
 *
 * Distinct from [CameraAccessOffScreen] because the remedy is different. Nothing the user can
 * change in Settings fixes this, so there is no "Open settings" button; what is offered instead
 * is the path that still works, which is picking an existing image.
 */
@Composable
fun CameraUnavailableScreen(
    onPickImage: () -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    ScreenColumn(modifier = modifier.background(Palm.Cam)) {
        Text(
            text = "The camera did not start",
            style = PalmText.ScreenH1,
            color = Color.White,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = "Another app may be using it. Close the other app and try again, or pick a " +
                "photo you have already taken.",
            style = PalmText.Body,
            color = Color.White.copy(alpha = 0.86f),
        )

        FlexibleSpace()

        OutlineButton(
            text = "Upload an image instead",
            onClick = onPickImage,
            onDark = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlineButton(
            text = "Close",
            onClick = onClose,
            onDark = true,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Preview(device = "spec:width=360dp,height=800dp")
@Composable
private fun CameraUnavailablePreview() {
    PalmRipenessTheme {
        CameraUnavailableScreen(onPickImage = {}, onClose = {})
    }
}

@Preview(device = "spec:width=360dp,height=800dp")
@Composable
private fun CameraAccessOffPreview() {
    PalmRipenessTheme {
        CameraAccessOffScreen(
            showReason = true,
            onOpenSettings = {},
            onPickImage = {},
            onClose = {},
        )
    }
}
