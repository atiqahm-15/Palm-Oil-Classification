package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import my.edu.um.palmripeness.ui.theme.Dim

/**
 * "Flexible space" from the handoff: a 16 dp floor plus the give that pushes the button block to
 * the bottom of a screen with room to spare. Once the content is taller than the screen the
 * weighted part collapses to nothing, the floor remains, and [ScreenColumn] scrolls.
 */
@Composable
fun ColumnScope.FlexibleSpace() {
    Spacer(Modifier.height(Dim.flexibleSpaceMin))
    Spacer(Modifier.weight(1f))
}

/**
 * A light screen with the handoff's padding.
 *
 * The content column is given the viewport as a *minimum* height inside a scroll container, which
 * is what lets [FlexibleSpace] distribute leftover space in the normal case and lets the screen
 * scroll instead of clipping at a 130% font scale — the text-scaling case the handoff calls out.
 */
@Composable
fun ScreenColumn(
    modifier: Modifier = Modifier,
    top: Dp = Dim.screenTop,
    bottom: Dp = Dim.screenBottom,
    content: @Composable ColumnScope.() -> Unit,
) {
    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val viewport = maxHeight
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState())) {
            Column(
                modifier = Modifier
                    .heightIn(min = viewport)
                    .fillMaxWidth()
                    .padding(
                        start = Dim.screenSide,
                        end = Dim.screenSide,
                        top = top,
                        bottom = bottom,
                    ),
                content = content,
            )
        }
    }
}
