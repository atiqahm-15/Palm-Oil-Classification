package my.edu.um.palmripeness.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

/**
 * One theme, no dark mode: the screens already split into light (Home, Results) and dark
 * (Camera, Review) by role, and a system-driven inversion would put the stage colours on
 * backgrounds they were never contrast-checked against.
 *
 * Material's scheme is filled in from the tokens so the few Material components in use
 * (`Text`, ripples) inherit the palette instead of the default purple.
 */
@Composable
fun PalmRipenessTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Palm.Ink,
            onPrimary = Palm.Ground,
            secondary = Palm.Ink2,
            onSecondary = Palm.Ground,
            background = Palm.Ground,
            onBackground = Palm.Ink,
            surface = Palm.Card,
            onSurface = Palm.Ink,
            surfaceVariant = Palm.Sand,
            onSurfaceVariant = Palm.Ink2,
            outline = Palm.Line,
            error = Palm.StageUnripe,
        ),
        typography = Typography(
            displayLarge = PalmText.Verdict,
            headlineLarge = PalmText.ScreenH1,
            headlineMedium = PalmText.HomeH1,
            titleLarge = PalmText.ReviewTitle,
            titleMedium = PalmText.Wordmark,
            bodyLarge = PalmText.Body,
            bodyMedium = PalmText.Notice,
            bodySmall = PalmText.Supporting,
            labelLarge = PalmText.PrimaryButton,
        ),
        content = content,
    )
}
