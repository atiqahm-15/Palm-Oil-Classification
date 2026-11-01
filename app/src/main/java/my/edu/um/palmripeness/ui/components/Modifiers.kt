package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import my.edu.um.palmripeness.ui.theme.Dim

/**
 * The focus treatment from the handoff: a 3 dp ring sitting 3 dp clear of the control.
 *
 * Drawn outside the node's own bounds rather than by padding the control inward, so a focused
 * button keeps the height the spec gives it (56 or 60 dp) instead of losing 12 dp to the ring.
 *
 * This does **not** add `focusable()` of its own. Every control that uses it goes on to add a
 * `clickable`, which is already a focus target, and adding a second one here made each button two
 * stops in the Tab order and left the ring watching whichever of the two happened to take focus.
 * `onFocusChanged` sits above the clickable in the chain, so it sees that node's focus directly.
 * A non-clickable control that ever wants this ring has to bring its own `focusable()`.
 *
 * @param cornerRadius the control's corner radius; for a circular control pass half its size.
 */
fun Modifier.focusRing(color: Color, cornerRadius: Dp): Modifier = composed {
    var focused by remember { mutableStateOf(false) }
    this
        .drawBehind {
            if (!focused) return@drawBehind
            val stroke = Dim.focusRingWidth.toPx()
            val inset = Dim.focusRingOffset.toPx() + stroke / 2f
            val radius = cornerRadius.toPx() + inset
            drawRoundRect(
                color = color,
                topLeft = Offset(-inset, -inset),
                size = Size(size.width + 2 * inset, size.height + 2 * inset),
                cornerRadius = CornerRadius(radius, radius),
                style = Stroke(width = stroke),
            )
        }
        .onFocusChanged { focused = it.isFocused }
}

/** Clickable with the project's ripple, and no click at all when disabled. */
@Composable
fun Modifier.tapTarget(
    onClick: () -> Unit,
    enabled: Boolean,
    interaction: MutableInteractionSource,
): Modifier = this.clickable(
    interactionSource = interaction,
    indication = ripple(),
    enabled = enabled,
    onClick = onClick,
)

/**
 * Clickable with a screen-reader label and an explicit role, for the icon-only camera controls:
 * the icon itself is decorative, so the label has to live on the touch target.
 */
@Composable
fun Modifier.clickableWithRole(
    onClick: () -> Unit,
    label: String,
    semanticRole: Role,
): Modifier {
    val interaction = remember { MutableInteractionSource() }
    return this
        .clickable(interactionSource = interaction, indication = ripple(), onClick = onClick)
        .semantics {
            contentDescription = label
            role = semanticRole
        }
}

/** A screen-reader label for a target whose click behaviour is already set up. */
fun Modifier.clickLabel(label: String): Modifier =
    this.semantics { contentDescription = label }

/** Radii the focus ring needs, named so they cannot drift from [Dim]'s shapes. */
object Radius {
    val button = 14.dp
    val gallery = 16.dp
}
