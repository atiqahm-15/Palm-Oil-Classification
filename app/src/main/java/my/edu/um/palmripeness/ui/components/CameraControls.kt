package my.edu.um.palmripeness.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import my.edu.um.palmripeness.ui.theme.InstrumentSans
import my.edu.um.palmripeness.ui.theme.Palm

/**
 * The camera screen's controls. Icons are custom and heavier than a stock outline set, per the
 * handoff; every control is at least 48 dp and carries its own screen-reader label, because none
 * of them has visible text.
 */

/** The round overlay buttons: close, and the flash toggle in its non-toggled form. */
@Composable
fun CameraRoundButton(
    iconRes: Int,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    diameter: Dp = 56.dp,
    fill: Color = Palm.Overlay,
    iconTint: Color = Color.White,
    role: Role = Role.Button,
) {
    Box(
        modifier = modifier
            .size(diameter)
            .focusRing(Palm.FocusDark, diameter / 2)
            .clip(CircleShape)
            .background(fill)
            .clickableWithRole(onClick, contentDescription, role),
        contentAlignment = Alignment.Center,
    ) {
        Image(
            painter = painterResource(iconRes),
            contentDescription = null, // the Box carries the label
            colorFilter = ColorFilter.tint(iconTint),
            modifier = Modifier.size(24.dp),
        )
    }
}

/**
 * The flash toggle. White fill with an ink icon when the torch is on, so its state is readable
 * without relying on the slash in the icon alone.
 */
@Composable
fun FlashToggle(
    on: Boolean,
    onIconRes: Int,
    offIconRes: Int,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
) {
    CameraRoundButton(
        iconRes = if (on) onIconRes else offIconRes,
        contentDescription = if (on) "Flash on" else "Flash off",
        onClick = onToggle,
        modifier = modifier,
        diameter = 48.dp,
        fill = if (on) Color.White else Palm.Overlay,
        iconTint = if (on) Palm.Ink else Color.White,
        role = Role.Switch,
    )
}

/** 84 dp: a 4 dp white ring around a 64 dp disc that shrinks to 56 dp while pressed. */
@Composable
fun Shutter(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    Box(
        modifier = modifier
            .size(84.dp)
            .focusRing(Palm.FocusDark, 42.dp)
            .clip(CircleShape)
            .border(4.dp, Color.White.copy(alpha = if (enabled) 1f else 0.5f), CircleShape)
            .tapTarget(onClick = onClick, enabled = enabled, interaction = interaction)
            .clickLabel("Take photo"),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(if (pressed) 56.dp else 64.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = if (enabled) 1f else 0.5f))
        )
    }
}

/** The gallery button: square-ish, 16 dp radius, a 2 dp white border at 75%. */
@Composable
fun GalleryButton(
    iconRes: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val shape = RoundedCornerShape(16.dp)
    Box(
        modifier = modifier
            .size(56.dp)
            .focusRing(Palm.FocusDark, Radius.gallery)
            .clip(shape)
            .background(Palm.Overlay)
            .border(2.dp, Color.White.copy(alpha = 0.75f), shape)
            .clickableWithRole(onClick, "Choose an image from the gallery", Role.Button),
        contentAlignment = Alignment.Center,
    ) {
        Image(
            painter = painterResource(iconRes),
            contentDescription = null,
            colorFilter = ColorFilter.tint(Color.White),
            modifier = Modifier.size(24.dp),
        )
    }
}

/** The advisory pill under the framing guide; its text changes in low light. */
@Composable
fun HintPill(text: String, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .widthIn(max = 300.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(Palm.OverlayStrong)
            .padding(vertical = 10.dp, horizontal = 16.dp),
    ) {
        Text(
            text = text,
            color = Color.White,
            textAlign = TextAlign.Center,
            fontFamily = InstrumentSans,
            fontWeight = FontWeight(500),
            fontSize = 15.sp,
        )
    }
}

/**
 * Four corner brackets on a 280 x 360 dp frame: 4 dp stroke, 40 dp arms, square caps.
 *
 * These bounds are load-bearing, not decoration: `CameraScreen` measures this composable against
 * the preview and crops the capture to it, so what the brackets hold is what the model is given.
 * Moving or resizing them changes the input the model sees — see `app/README.md`.
 */
@Composable
fun FramingGuide(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier.size(width = 280.dp, height = 360.dp)) {
        val stroke = 4.dp.toPx()
        val arm = 40.dp.toPx()
        val inset = stroke / 2f
        val right = size.width - inset
        val bottom = size.height - inset

        fun bracket(x: Float, y: Float, dx: Float, dy: Float) {
            drawLine(Color.White, Offset(x, y), Offset(x + dx, y), stroke, StrokeCap.Butt)
            drawLine(Color.White, Offset(x, y), Offset(x, y + dy), stroke, StrokeCap.Butt)
        }
        bracket(inset, inset, arm, arm)
        bracket(right, inset, -arm, arm)
        bracket(inset, bottom, arm, -arm)
        bracket(right, bottom, -arm, -arm)
    }
}
