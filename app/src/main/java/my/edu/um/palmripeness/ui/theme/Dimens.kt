package my.edu.um.palmripeness.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp

/** Spacing and shape from the handoff table. Reference screen is 360 x 800 dp. */
object Dim {
    val screenSide = 20.dp
    val screenTop = 52.dp
    val screenBottom = 20.dp
    val homeBottom = 24.dp
    val cameraControlsBottom = 40.dp

    /**
     * Space at the bottom of the camera preview that the framing guide and its hint must stay
     * clear of: the 84 dp shutter plus [cameraControlsBottom] plus breathing room. The guide
     * block is centred in what is left, so it cannot land under the controls on a short screen.
     */
    val cameraGuideReserve = 140.dp

    /** The floor a "flexible space" spacer shrinks to before the screen starts scrolling. */
    val flexibleSpaceMin = 16.dp

    val buttonShape = RoundedCornerShape(14.dp)
    val cardShape = RoundedCornerShape(20.dp)
    val homeStageCardShape = RoundedCornerShape(24.dp)
    val noticeShape = RoundedCornerShape(14.dp)
    val chartTrackShape = RoundedCornerShape(8.dp)
    val galleryButtonShape = RoundedCornerShape(16.dp)

    val primaryButtonHeight = 56.dp
    val primaryButtonTallHeight = 60.dp
    val outlineButtonHeight = 56.dp

    val focusRingWidth = 3.dp
    val focusRingOffset = 3.dp
}
