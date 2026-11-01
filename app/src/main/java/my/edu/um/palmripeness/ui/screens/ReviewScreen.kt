package my.edu.um.palmripeness.ui.screens

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.ui.components.OutlineButton
import my.edu.um.palmripeness.ui.components.PrimaryButton
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * Review, the Preview step in the report. One decision: is this photo good enough to spend an
 * inference on, or should it be retaken?
 *
 * The image is fitted rather than cropped, so what is shown is what the model will be given — a
 * crop here would hide the framing problem the screen exists to catch.
 *
 * The photo area takes the space left over above the buttons rather than the flat 640 dp the
 * handoff specifies. On the 360 x 800 reference those are the same thing, but a fixed height
 * leaves a dead band on a taller phone, and every phone this is meant for is taller.
 */
@Composable
fun ReviewScreen(
    photo: Bitmap?,
    onRetake: () -> Unit,
    onDetect: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.fillMaxSize().background(Palm.Cam)) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Palm.CamPhoto),
            contentAlignment = Alignment.Center,
        ) {
            if (photo != null) {
                Image(
                    bitmap = photo.asImageBitmap(),
                    contentDescription = "The photo you just took",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            // The title sits on the photograph, and a bunch in sunlight is bright enough to lose
            // white text against it. This scrim is the smallest thing that keeps the copy legible
            // on any photo without putting a panel over the image.
            //
            // It holds 0.62 flat for the height the two lines occupy and only fades below them.
            // The previous version faded from the very top, which left the subtitle sitting where
            // the scrim had already dropped to roughly 0.3 — enough to fail 4.5:1 against a
            // sunlit bunch, which is exactly the photo this screen is for.
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth()
                    .height(260.dp)
                    .background(
                        Brush.verticalGradient(
                            0.0f to Color.Black.copy(alpha = 0.62f),
                            0.62f to Color.Black.copy(alpha = 0.62f),
                            1.0f to Color.Transparent,
                        )
                    )
            )

            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .windowInsetsPadding(WindowInsets.statusBars)
                    .padding(start = Dim.screenSide, end = Dim.screenSide, top = 52.dp),
            ) {
                Text(text = "Check the photo", style = PalmText.ReviewTitle, color = Color.White)
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "Is the bunch sharp and fully in frame?",
                    style = PalmText.Body,
                    color = Color.White.copy(alpha = 0.86f),
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(
                    start = Dim.screenSide,
                    end = Dim.screenSide,
                    top = 20.dp,
                    bottom = Dim.cameraControlsBottom,
                ),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlineButton(
                text = "Retake",
                onClick = onRetake,
                onDark = true,
                height = Dim.primaryButtonTallHeight,
                modifier = Modifier.weight(1f),
            )
            PrimaryButton(
                text = "Detect ripeness",
                onClick = onDetect,
                enabled = photo != null,
                onDark = true,
                height = Dim.primaryButtonTallHeight,
                modifier = Modifier.weight(2f),
            )
        }
    }
}
