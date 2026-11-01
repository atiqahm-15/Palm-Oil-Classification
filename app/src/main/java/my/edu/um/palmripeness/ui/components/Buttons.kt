package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.ui.theme.Dim
import my.edu.um.palmripeness.ui.theme.Palm
import my.edu.um.palmripeness.ui.theme.PalmText

/**
 * The two text buttons from the handoff's Components table. Built from primitives rather than
 * Material's `Button` so the pressed, disabled and focus treatments are the specified ones and
 * no Material default leaks in.
 */

/**
 * ink fill, ground text. Pressed darkens the fill 10%.
 *
 * On the dark screens the fill and text swap: Review puts this button on `cam`, where ink on ink
 * would disappear, so it becomes ground fill with ink text.
 */
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    onDark: Boolean = false,
    height: Dp = Dim.primaryButtonHeight,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val base = if (onDark) Palm.Ground else Palm.Ink
    val fill = when {
        !enabled -> Palm.DisabledFill
        pressed -> lerp(base, Color.Black, 0.10f)
        else -> base
    }
    val content = when {
        !enabled -> Palm.DisabledText
        onDark -> Palm.Ink
        else -> Palm.Ground
    }
    Box(
        modifier = modifier
            .height(height)
            .focusRing(if (onDark) Palm.FocusDark else Palm.Ink, Radius.button)
            .clip(Dim.buttonShape)
            .background(fill)
            .tapTarget(onClick = onClick, enabled = enabled, interaction = interaction)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text = text, style = PalmText.PrimaryButton, color = content)
    }
}

/**
 * Transparent with a 2 dp border. On the dark screens the border and text are white and the
 * focus ring is focus-dark, per the handoff.
 */
@Composable
fun OutlineButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    onDark: Boolean = false,
    height: Dp = Dim.outlineButtonHeight,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val content = when {
        !enabled -> Palm.DisabledText
        onDark -> Color.White
        else -> Palm.Ink
    }
    val fill = when {
        !enabled -> Palm.DisabledFill
        pressed && onDark -> Color.White.copy(alpha = 0.08f)
        pressed -> Palm.Ink.copy(alpha = 0.08f)
        else -> Color.Transparent
    }
    Box(
        modifier = modifier
            .height(height)
            .focusRing(if (onDark) Palm.FocusDark else Palm.Ink, Radius.button)
            .clip(Dim.buttonShape)
            .background(fill)
            .border(BorderStroke(2.dp, content), Dim.buttonShape)
            .tapTarget(onClick = onClick, enabled = enabled, interaction = interaction)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text = text, style = PalmText.OutlineButton, color = content)
    }
}
