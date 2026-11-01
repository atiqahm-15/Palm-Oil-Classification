package my.edu.um.palmripeness.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.MainViewModel.ModelState
import my.edu.um.palmripeness.R
import my.edu.um.palmripeness.ui.components.FlexibleSpace
import my.edu.um.palmripeness.ui.components.NoticeBox
import my.edu.um.palmripeness.ui.components.NoticeTone
import my.edu.um.palmripeness.ui.components.OutlineButton
import my.edu.um.palmripeness.ui.components.PrimaryButton
import my.edu.um.palmripeness.ui.components.ScreenColumn
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmRipenessTheme
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * Home. Two ways in — camera or the system picker — and a strip naming the four stages the model
 * tells apart, so the vocabulary on the Results screen is not new when it first appears.
 *
 * When the interpreter failed to load there is nothing this screen can usefully offer, so both
 * buttons are disabled and the notice says what to do about it.
 */
@Composable
fun HomeScreen(
    modelState: ModelState,
    stageLabels: List<String>,
    onTakePhoto: () -> Unit,
    onUploadImage: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val ready = modelState is ModelState.Ready

    ScreenColumn(
        modifier = modifier.background(Palm.Ground),
        bottom = Dim.homeBottom,
    ) {
        Text(
            text = "Palm Ripeness",
            style = PalmText.Wordmark,
            color = Palm.Ink,
            modifier = Modifier.height(32.dp),
        )

        Spacer(Modifier.height(20.dp))
        StageStrip(stageLabels)

        Spacer(Modifier.height(24.dp))
        Text(
            text = "Which stage is this bunch in?",
            style = PalmText.HomeH1,
            color = Palm.Ink,
        )
        Spacer(Modifier.height(10.dp))
        Text(
            text = "Photograph one fresh fruit bunch and the app names its ripeness stage. " +
                "Everything runs on this phone.",
            style = PalmText.Body,
            color = Palm.Ink2,
        )

        FlexibleSpace()

        if (modelState is ModelState.Failed) {
            NoticeBox(
                text = "The model did not load. Close the app and open it again. Photos cannot " +
                    "be analysed until it loads.",
                tone = NoticeTone.Error,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(12.dp))
        }

        PrimaryButton(
            text = "Take photo",
            onClick = onTakePhoto,
            enabled = ready,
            height = Dim.primaryButtonTallHeight,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlineButton(
            text = "Upload image",
            onClick = onUploadImage,
            enabled = ready,
            modifier = Modifier.fillMaxWidth(),
        )

        Spacer(Modifier.height(16.dp))
        Text(
            text = when (modelState) {
                is ModelState.Ready -> "Model ready · works offline"
                is ModelState.Failed -> "Model not loaded"
                ModelState.Loading -> "Loading model"
            },
            style = PalmText.Supporting,
            color = Palm.Ink2,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

/**
 * The four stages, in ripeness order, left to right — the same order the Results chart uses, so
 * the reading stays consistent for anyone who cannot separate the red from the green.
 *
 * The drawings stand in for real photos; see "Placeholders to replace" in DESIGN_NOTES.md.
 */
@Composable
private fun StageStrip(stageLabels: List<String>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(Dim.homeStageCardShape)
            .background(Palm.Sand)
            .padding(horizontal = 14.dp, vertical = 12.dp),
    ) {
        Text(
            text = "The four stages it tells apart",
            style = PalmText.Supporting,
            color = Palm.Ink2,
        )
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            stageLabels.forEach { label ->
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Image(
                        painter = painterResource(R.drawable.ic_bunch_silhouette),
                        contentDescription = null, // decorative: the name below carries the meaning
                        // Modulate, not the default SrcIn: SrcIn would repaint every pixel the same
                        // colour and flatten the drupe shading out of the drawing.
                        colorFilter = ColorFilter.tint(
                            Palm.stageColor(label), BlendMode.Modulate
                        ),
                        modifier = Modifier.size(width = 64.dp, height = 77.dp),
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = Palm.stageDisplayName(label),
                        style = PalmText.StageLabel,
                        color = Palm.Ink,
                        textAlign = TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Preview(device = "spec:width=360dp,height=800dp", showBackground = true)
@Composable
private fun HomeLoadingPreview() {
    PalmRipenessTheme {
        HomeScreen(
            modelState = ModelState.Loading,
            stageLabels = listOf("unripe", "underripe", "ripe", "overripe"),
            onTakePhoto = {},
            onUploadImage = {},
        )
    }
}

@Preview(device = "spec:width=360dp,height=800dp", showBackground = true)
@Composable
private fun HomeFailedPreview() {
    PalmRipenessTheme {
        HomeScreen(
            modelState = ModelState.Failed("model.tflite missing"),
            stageLabels = listOf("unripe", "underripe", "ripe", "overripe"),
            onTakePhoto = {},
            onUploadImage = {},
        )
    }
}
