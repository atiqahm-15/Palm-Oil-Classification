package my.edu.um.palmripeness.ui.theme

import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import my.edu.um.palmripeness.R

/**
 * Both faces are bundled in `res/font` because the app requests no network permission, so a
 * downloadable-font provider would never resolve.
 *
 * They ship as variable fonts (the only form Google Fonts publishes them in), so the weight has
 * to be set on the `wght` axis as well as declared as a [FontWeight] — Fraunces' default
 * instance is Thin, and a family that only declares the weight renders hairline text. Variation
 * settings need API 26, which is where `minSdk` comes from.
 */
@OptIn(ExperimentalTextApi::class)
private fun fraunces(weight: Int, opticalSize: Float) = FontFamily(
    Font(
        R.font.fraunces,
        weight = FontWeight(weight),
        variationSettings = FontVariation.Settings(
            FontVariation.weight(weight),
            FontVariation.Setting("opsz", opticalSize),
        ),
    )
)

@OptIn(ExperimentalTextApi::class)
private fun instrumentSans(weight: Int) = Font(
    R.font.instrument_sans,
    weight = FontWeight(weight),
    variationSettings = FontVariation.Settings(FontVariation.weight(weight)),
)

/** Fraunces at a display optical size, for the 54 sp verdict only. */
val FrauncesDisplay = fraunces(weight = 600, opticalSize = 96f)

/** Fraunces at a heading optical size, for every other Fraunces style. */
val FrauncesHeading = fraunces(weight = 600, opticalSize = 30f)

val InstrumentSans = FontFamily(
    instrumentSans(400),
    instrumentSans(500),
    instrumentSans(600),
)

/**
 * The text styles from the handoff table. Named after the row there rather than after a
 * Material slot, so a spec change has exactly one place to land.
 */
object PalmText {
    val Verdict = TextStyle(
        fontFamily = FrauncesDisplay, fontWeight = FontWeight(600),
        fontSize = 54.sp, lineHeight = 54.sp,
    )
    val ScreenH1 = TextStyle(
        fontFamily = FrauncesHeading, fontWeight = FontWeight(600),
        fontSize = 34.sp, lineHeight = 37.4.sp,
    )
    val HomeH1 = TextStyle(
        fontFamily = FrauncesHeading, fontWeight = FontWeight(600),
        fontSize = 30.sp, lineHeight = 34.5.sp,
    )
    val ReviewTitle = TextStyle(
        fontFamily = FrauncesHeading, fontWeight = FontWeight(600), fontSize = 24.sp,
    )
    val Wordmark = TextStyle(
        fontFamily = FrauncesHeading, fontWeight = FontWeight(600), fontSize = 22.sp,
    )
    val Body = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(400),
        fontSize = 16.sp, lineHeight = 24.sp,
    )
    val PrimaryButton = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(600), fontSize = 18.sp,
        textAlign = TextAlign.Center,
    )
    val OutlineButton = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(600), fontSize = 17.sp,
        textAlign = TextAlign.Center,
    )
    val Confidence = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(400),
        fontSize = 18.sp, lineHeight = 23.4.sp,
    )
    val ChartValue = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(400), fontSize = 15.sp,
    )
    val Notice = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(400),
        fontSize = 15.sp, lineHeight = 21.sp,
    )
    val Supporting = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(400), fontSize = 14.sp,
    )
    val StageLabel = TextStyle(
        fontFamily = InstrumentSans, fontWeight = FontWeight(600), fontSize = 14.sp,
    )
}
