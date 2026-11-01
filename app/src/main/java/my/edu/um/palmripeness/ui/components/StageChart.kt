package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * Model confidence by stage: one column per class, in ripeness order, left to right.
 *
 * The chart exists so a close call is visible rather than inferred — 51% ripe against 45%
 * underripe should not look like a confident answer. Column heights are therefore true, with no
 * minimum: a 0.3% score is drawn as very nearly nothing.
 */
@Composable
fun StageChart(
    labels: List<String>,
    scores: FloatArray,
    topIndex: Int,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(Dim.cardShape)
            .background(Palm.Card)
            .border(1.dp, Palm.Line, Dim.cardShape)
            .padding(14.dp),
    ) {
        Text(
            text = "Model confidence by stage",
            style = PalmText.ChartValue.copy(fontWeight = FontWeight(600)),
            color = Palm.Ink,
        )
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            labels.forEachIndexed { index, label ->
                StageColumn(
                    label = label,
                    score = scores.getOrElse(index) { 0f },
                    isTop = index == topIndex,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

@Composable
private fun StageColumn(
    label: String,
    score: Float,
    isTop: Boolean,
    modifier: Modifier = Modifier,
) {
    val percent = score * 100f
    val weight = if (isTop) FontWeight(600) else FontWeight(400)
    Column(
        modifier = modifier.clearAndSetSemantics {
            // One announcement per column, in the order a sighted reader takes it in.
            contentDescription = "${Palm.stageDisplayName(label)}, " +
                String.format("%.1f", percent) + " percent"
        },
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = String.format("%.1f%%", percent),
            style = PalmText.ChartValue.copy(fontWeight = weight),
            color = Palm.Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(6.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(96.dp)
                .clip(Dim.chartTrackShape)
                .background(Palm.Track),
            contentAlignment = Alignment.BottomCenter,
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(fraction = score.coerceIn(0f, 1f))
                    .clip(Dim.chartTrackShape)
                    .background(Palm.stageColor(label))
            )
        }
        Spacer(Modifier.height(6.dp))
        Text(
            text = Palm.stageDisplayName(label),
            style = PalmText.Supporting.copy(fontWeight = weight),
            color = Palm.Ink,
            textAlign = TextAlign.Center,
        )
    }
}
