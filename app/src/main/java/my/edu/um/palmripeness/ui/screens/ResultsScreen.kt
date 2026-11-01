package my.edu.um.palmripeness.ui.screens

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.MainViewModel.Analysis
import my.edu.um.palmripeness.model.Prediction
import my.edu.um.palmripeness.ui.components.FlexibleSpace
import my.edu.um.palmripeness.ui.components.NoticeBox
import my.edu.um.palmripeness.ui.components.NoticeTone
import my.edu.um.palmripeness.ui.components.OutlineButton
import my.edu.um.palmripeness.ui.components.PrimaryButton
import my.edu.um.palmripeness.ui.components.ScreenColumn
import my.edu.um.palmripeness.ui.components.StageChart
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmRipenessTheme
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * Results, in its three states: analysing, a verdict, or a photo that could not be read.
 *
 * The number is labelled "model confidence" throughout, never "probability of correctness": the
 * exported model is uncalibrated, as `model_contract.json` records, and the report says so.
 */
@Composable
fun ResultsScreen(
    analysis: Analysis,
    photo: Bitmap?,
    onNewPhoto: () -> Unit,
    onDone: () -> Unit,
    onCancel: () -> Unit,
    onChooseAnother: () -> Unit,
    modifier: Modifier = Modifier,
) {
    when (analysis) {
        Analysis.Running, Analysis.Idle -> AnalysingState(onCancel, modifier)
        Analysis.Unreadable -> UnreadableState(onNewPhoto, onChooseAnother, modifier)
        is Analysis.Done -> VerdictState(analysis.prediction, photo, onNewPhoto, onDone, modifier)
    }
}

/**
 * Inference takes tens of milliseconds, so there is no progress to report and the screen shows no
 * percentage. It is held for a short floor (see `MainViewModel.MIN_ANALYSING_MILLIS`) so the step
 * registers as a step instead of a flicker.
 */
@Composable
private fun AnalysingState(onCancel: () -> Unit, modifier: Modifier) {
    ScreenColumn(modifier = modifier.background(Palm.Ground)) {
        Text(text = "Analysing photo", style = PalmText.ScreenH1, color = Palm.Ink)
        Spacer(Modifier.height(16.dp))
        CircularProgressIndicator(color = Palm.Ink, strokeWidth = 3.dp)
        Spacer(Modifier.height(16.dp))
        Text(
            text = "The model runs on your phone, so this works without signal. The photo is " +
                "not uploaded.",
            style = PalmText.Body,
            color = Palm.Ink2,
        )
        FlexibleSpace()
        OutlineButton(
            text = "Cancel",
            onClick = onCancel,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun UnreadableState(
    onNewPhoto: () -> Unit,
    onChooseAnother: () -> Unit,
    modifier: Modifier,
) {
    ScreenColumn(modifier = modifier.background(Palm.Ground)) {
        Text(
            text = "This photo could not be read",
            style = PalmText.ScreenH1,
            color = Palm.Ink,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = "The image may be blurred, too dark, or in a format the app cannot open. " +
                "Take another photo with the whole bunch in frame.",
            style = PalmText.Body,
            color = Palm.Ink2,
        )
        FlexibleSpace()
        PrimaryButton(
            text = "Take another photo",
            onClick = onNewPhoto,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlineButton(
            text = "Choose another image",
            onClick = onChooseAnother,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun VerdictState(
    prediction: Prediction,
    photo: Bitmap?,
    onNewPhoto: () -> Unit,
    onDone: () -> Unit,
    modifier: Modifier,
) {
    val stage = Palm.stageDisplayName(prediction.topLabel)
    val confidence = prediction.percent(prediction.topIndex)

    ScreenColumn(modifier = modifier.background(Palm.Ground)) {
        Text(
            text = "Result",
            style = PalmText.Wordmark,
            color = Palm.Ink,
            modifier = Modifier.height(36.dp),
        )

        Spacer(Modifier.height(12.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(152.dp)
                .clip(Dim.cardShape)
                .background(Palm.Cam),
            contentAlignment = Alignment.Center,
        ) {
            if (photo != null) {
                Image(
                    bitmap = photo.asImageBitmap(),
                    contentDescription = "The photo that was analysed",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                // Reachable when the process was killed after the verdict and the bitmap was not
                // kept (see MainViewModel). The verdict below is still valid, so the frame says
                // what is missing instead of showing an unexplained empty rectangle.
                Text(
                    text = "Photo not kept",
                    style = PalmText.Supporting,
                    color = Color.White.copy(alpha = 0.7f),
                )
            }
        }

        Spacer(Modifier.height(20.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.clearAndSetSemantics {
                contentDescription = "$stage, model confidence " +
                    confidence.removeSuffix("%") + " percent"
            },
        ) {
            Box(
                Modifier
                    .size(20.dp)
                    .clip(CircleShape)
                    .background(Palm.stageColor(prediction.topLabel))
            )
            Spacer(Modifier.width(12.dp))
            Text(text = stage, style = PalmText.Verdict, color = Palm.Ink)
        }
        Spacer(Modifier.height(6.dp))
        Text(
            text = "Model confidence $confidence",
            style = PalmText.Confidence,
            color = Palm.Ink2,
        )

        if (prediction.isCloseCall) {
            Spacer(Modifier.height(12.dp))
            NoticeBox(
                text = "Close call. ${prediction.percent(prediction.secondIndex)} " +
                    "${prediction.secondLabel}. Check the bunch by eye.",
                tone = NoticeTone.CloseCall,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Spacer(Modifier.height(20.dp))
        StageChart(
            labels = prediction.labels,
            scores = prediction.scores,
            topIndex = prediction.topIndex,
        )

        FlexibleSpace()
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlineButton(
                text = "Done",
                onClick = onDone,
                modifier = Modifier.weight(1f),
            )
            PrimaryButton(
                text = "New photo",
                onClick = onNewPhoto,
                modifier = Modifier.weight(2f),
            )
        }
        Spacer(Modifier.height(12.dp))
        Text(
            text = "Analysed on this phone. Nothing was uploaded.",
            style = PalmText.Supporting,
            color = Palm.Ink2,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private val previewLabels = listOf("unripe", "underripe", "ripe", "overripe")

@Preview(device = "spec:width=360dp,height=800dp", showBackground = true)
@Composable
private fun ResultsConfidentPreview() {
    PalmRipenessTheme {
        ResultsScreen(
            analysis = Analysis.Done(
                Prediction(previewLabels, floatArrayOf(0.007f, 0.021f, 0.964f, 0.008f), 47L)
            ),
            photo = null,
            onNewPhoto = {}, onDone = {}, onCancel = {}, onChooseAnother = {},
        )
    }
}

@Preview(device = "spec:width=360dp,height=800dp", showBackground = true)
@Composable
private fun ResultsCloseCallPreview() {
    PalmRipenessTheme {
        ResultsScreen(
            analysis = Analysis.Done(
                Prediction(previewLabels, floatArrayOf(0.02f, 0.45f, 0.51f, 0.02f), 47L)
            ),
            photo = null,
            onNewPhoto = {}, onDone = {}, onCancel = {}, onChooseAnother = {},
        )
    }
}

@Preview(device = "spec:width=360dp,height=800dp", showBackground = true)
@Composable
private fun ResultsUnreadablePreview() {
    PalmRipenessTheme {
        ResultsScreen(
            analysis = Analysis.Unreadable,
            photo = null,
            onNewPhoto = {}, onDone = {}, onCancel = {}, onChooseAnother = {},
        )
    }
}
