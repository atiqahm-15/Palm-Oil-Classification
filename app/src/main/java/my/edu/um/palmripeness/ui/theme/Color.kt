package my.edu.um.palmripeness.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * The design tokens from `app/DESIGN_HANDOFF.md`, one-to-one with the table there.
 *
 * Green means ripe and nothing else, so no button or accent uses [StageRipe] or any other green.
 */
object Palm {
    val Ground = Color(0xFFF3EEE2)
    val Card = Color(0xFFFFFCF5)
    val Sand = Color(0xFFE7DEC6)
    val Ink = Color(0xFF1A271D)
    val Ink2 = Color(0xFF4B584D)
    val Track = Color(0xFFE6DEC9)
    val Line = Color(0xFFE2D9C4)
    val Cam = Color(0xFF131C16)
    val CamPhoto = Color(0xFF1B271F)

    val StageUnripe = Color(0xFFB3261E)
    val StageUnderripe = Color(0xFFA86A00)
    val StageRipe = Color(0xFF2F7A3E)
    val StageOverripe = Color(0xFF4A3428)

    val FocusDark = Color(0xFFF0B429)

    val DisabledFill = Color(0xFFD9D1BC)
    val DisabledText = Color(0xFF5A655C)

    val Overlay = Color(0xFF0B100C).copy(alpha = 0.62f)
    val OverlayStrong = Color(0xFF0B100C).copy(alpha = 0.78f)

    /**
     * Stage colour by class name. The names come from `labels.txt` at runtime, never from an
     * enum ordinal, so a re-export that reorders the classes cannot silently recolour the chart.
     */
    fun stageColor(label: String): Color = when (label.lowercase()) {
        "unripe" -> StageUnripe
        "underripe" -> StageUnderripe
        "ripe" -> StageRipe
        "overripe" -> StageOverripe
        else -> Ink2
    }

    /** Sentence-case name for display; the bundled labels are lowercase. */
    fun stageDisplayName(label: String): String =
        label.replaceFirstChar { it.uppercase() }
}
