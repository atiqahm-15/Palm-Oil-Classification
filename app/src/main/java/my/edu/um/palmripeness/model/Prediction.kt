package my.edu.um.palmripeness.model

/**
 * One inference: the four softmax scores in `labels.txt` order, and how long the interpreter
 * took. The latency is kept because the ≤ 2 s on-device requirement has to be evidenced from a
 * real phone, not from the conversion spike's 47.2 ms.
 */
data class Prediction(
    val labels: List<String>,
    val scores: FloatArray,
    val inferenceMillis: Long,
) {
    init {
        require(labels.size == scores.size) { "${labels.size} labels but ${scores.size} scores" }
    }

    /** Indices ordered by score, highest first. */
    private val ranking: List<Int> = scores.indices.sortedByDescending { scores[it] }

    val topIndex: Int get() = ranking.first()
    val topLabel: String get() = labels[topIndex]
    val topScore: Float get() = scores[topIndex]

    val secondIndex: Int get() = ranking[1]
    val secondLabel: String get() = labels[secondIndex]
    val secondScore: Float get() = scores[secondIndex]

    /**
     * The close-call notice fires when the top two scores are less than [MARGIN_POINTS]
     * percentage points apart. The threshold is a design assumption, not a number taken from
     * the validation results — see the open question in `app/DESIGN_HANDOFF.md`.
     */
    val isCloseCall: Boolean get() = (topScore - secondScore) * 100f < MARGIN_POINTS

    /** Score as a percentage to one decimal place, e.g. "96.4%". */
    fun percent(index: Int): String = String.format("%.1f%%", scores[index] * 100f)

    override fun equals(other: Any?): Boolean =
        other is Prediction && labels == other.labels &&
            scores.contentEquals(other.scores) && inferenceMillis == other.inferenceMillis

    override fun hashCode(): Int =
        (labels.hashCode() * 31 + scores.contentHashCode()) * 31 + inferenceMillis.hashCode()

    companion object {
        const val MARGIN_POINTS = 15f
    }
}
