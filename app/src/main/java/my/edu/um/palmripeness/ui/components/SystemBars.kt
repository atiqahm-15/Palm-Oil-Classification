package my.edu.um.palmripeness.ui.components

import android.app.Activity
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

/**
 * Matches the system bar icons to the screen underneath them.
 *
 * The app draws edge to edge, and its screens are not all the same brightness: Home and Results
 * are `ground`, Camera and Review are `cam`. The window theme can only state one preference, and
 * it states the light one, so without this the two dark screens get dark status-bar icons on a
 * near-black background and the clock and battery disappear.
 *
 * @param darkIcons true for a light background (dark icons), false for a dark one.
 */
@Composable
fun SystemBarIcons(darkIcons: Boolean) {
    val view = LocalView.current
    if (view.isInEditMode) return
    SideEffect {
        val window = (view.context as Activity).window
        WindowCompat.getInsetsController(window, view).apply {
            isAppearanceLightStatusBars = darkIcons
            isAppearanceLightNavigationBars = darkIcons
        }
    }
}
