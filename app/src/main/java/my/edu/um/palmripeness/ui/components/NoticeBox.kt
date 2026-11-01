package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.R
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmText

/** Which of the two notice treatments to use. */
enum class NoticeTone {
    /** Ink fill, ground text — errors and blocking conditions. */
    Error,

    /** Sand fill, ink text — the close-call notice on Results. */
    CloseCall,
}

/**
 * The Notice box from the handoff's Components table: radius 14, padding 14/16, 15 sp copy, and
 * a 22 dp filled info icon at the left.
 */
@Composable
fun NoticeBox(
    text: String,
    tone: NoticeTone,
    modifier: Modifier = Modifier,
) {
    val fill = if (tone == NoticeTone.Error) Palm.Ink else Palm.Sand
    val content = if (tone == NoticeTone.Error) Palm.Ground else Palm.Ink
    Row(
        modifier = modifier
            .clip(Dim.noticeShape)
            .background(fill)
            .padding(vertical = 14.dp, horizontal = 16.dp)
            // The copy already says everything; the icon must not be read out twice.
            .semantics { },
        verticalAlignment = Alignment.Top,
    ) {
        Image(
            painter = painterResource(R.drawable.ic_info_filled),
            contentDescription = null,
            colorFilter = ColorFilter.tint(content),
            modifier = Modifier.size(22.dp),
        )
        Spacer(Modifier.width(12.dp))
        Text(text = text, style = PalmText.Notice, color = content)
    }
}
