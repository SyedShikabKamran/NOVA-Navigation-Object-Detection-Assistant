package com.nova.assistant.features.voicecommand

/**
 * VoiceMatcher — scored intent matcher that replaces the old substring `contains()` ladder.
 *
 * WHY: the previous `matchCommand` did `text.contains("stop")` etc. with manual priority
 * ordering. That fired commands on ambient speech ("I need to STOP at the pharmacy" → PAUSE;
 * "that was an EMERGENCY earlier" → SOS) and had zero tolerance for ASR mishears.
 *
 * HOW: every command has trigger phrases. For an utterance we compute, per command:
 *   phraseSim — how well the best-aligned token window matches a trigger phrase
 *               (Jaro-Winkler per token, tolerant of mishears: "cher"≈"chair").
 *   coverage  — fraction of the UTTERANCE's tokens that the trigger covers.
 * score = phraseSim * (ALPHA + (1-ALPHA)*coverage) * rankPrior.
 *
 * Coverage is the key: a 7-word overheard sentence containing "stop" has tiny coverage of
 * PAUSE's "stop" → low score → rejected. A crisp "pause" has coverage 1.0. Per-command
 * thresholds let short/dangerous commands (EMERGENCY, PAUSE) demand a near-exact match.
 *
 * This object is pure Kotlin (no Android imports) so it is unit-tested on the JVM
 * (VoiceMatcherTest) — important because the matcher is a safety path.
 *
 * Calibration: thresholds below are validated against a labelled prototype (ambient speech
 * rejected, real commands + mishears accepted). They are STARTING points — tune on real
 * logged `ASR_HEARD` transcripts (FileLogger) against the FAR/FRR curve for NOVA's users.
 *
 * Deliberately NOT done here (see PROGRESS.md): Double Metaphone phonetic keys (Jaro-Winkler
 * covers most sound-alikes; add Apache Commons Codec if field tests show mishear misses),
 * the confirm/countdown dialog state machine, and the n-best confidence weighting (offline
 * returns no confidence and usually a single hypothesis).
 */

/** Cost-of-error class. Drives the (deferred) act/confirm/countdown policy; risk is data now. */
enum class Risk { CHEAP, NAVIGATIONAL, DESTRUCTIVE, SAFETY, SAFETY_CANCEL }

/** GLOBAL = always matchable. FREETEXT_SUPPRESSED = skip while a free-text capture is active. */
enum class ContextScope { GLOBAL, FREETEXT_SUPPRESSED }

internal const val MATCH_ALPHA = 0.6        // phraseSim floor so a crisp phrase isn't over-penalised
internal const val GLOBAL_ACCEPT = 0.80     // default accept threshold; per-command overrides below

/** Lowercase, strip punctuation, split into alphanumeric tokens. */
internal fun tokenize(s: String): List<String> =
    Regex("[a-z0-9]+").findAll(s.lowercase()).map { it.value }.toList()

data class CommandSpec(
    val command: NovaCommand,
    val phrases: List<String>,           // trigger phrasings / aliases — add a row to extend
    val risk: Risk,
    val threshold: Double = GLOBAL_ACCEPT,
    val requiresConfirm: Boolean = false, // reserved for the deferred confirm dialog
    val cooldownExempt: Boolean = false,
    val contextScope: ContextScope = ContextScope.GLOBAL,
) {
    val phraseTokens: List<List<String>> = phrases.map { tokenize(it) }
    val canonical: String get() = phrases.first()
}

data class MatchResult(
    val spec: CommandSpec?,
    val score: Double,
    val accepted: Boolean,
) {
    val command: NovaCommand? get() = spec?.command
}

object VoiceMatcher {

    fun match(nbest: List<String>, freeTextMode: Boolean = false): MatchResult {
        var best: CommandSpec? = null
        var bestScore = 0.0
        for (spec in CommandCatalog.commands) {
            // Free-text capture (room name / number) suppresses general commands, but never
            // the safety command — the user must never be trapped unable to call for help.
            if (freeTextMode && spec.contextScope == ContextScope.FREETEXT_SUPPRESSED &&
                spec.risk != Risk.SAFETY
            ) continue

            var cmdBest = 0.0
            nbest.forEachIndexed { i, hypothesis ->
                val utt = tokenize(hypothesis)
                if (utt.isNotEmpty()) {
                    // Top hypothesis (rank 0) prior = 1.0 — the calibrated path. Lower hypotheses
                    // are penalised so they can't manufacture a false positive (validated: n-best
                    // never resurrects a rejected ambient command).
                    val prior = 1.0 / (1.0 + 0.5 * i)
                    for (phrase in spec.phraseTokens) {
                        val (sim, cov) = phraseScore(phrase, utt)
                        val s = sim * (MATCH_ALPHA + (1.0 - MATCH_ALPHA) * cov) * prior
                        if (s > cmdBest) cmdBest = s
                    }
                }
            }
            if (cmdBest > bestScore) {
                bestScore = cmdBest
                best = spec
            }
        }
        val threshold = best?.threshold ?: GLOBAL_ACCEPT
        return MatchResult(best, bestScore, best != null && bestScore >= threshold)
    }

    /**
     * Slide the trigger phrase over the utterance's token windows; return the best window's
     * (phraseSim, coverage). Greedy per-token alignment is fine for <=4-token triggers.
     */
    internal fun phraseScore(phrase: List<String>, utt: List<String>): Pair<Double, Double> {
        if (phrase.isEmpty() || utt.isEmpty()) return 0.0 to 0.0
        val n = phrase.size
        var bestSim = 0.0
        var bestUsed = 0
        val lastStart = maxOf(1, utt.size - n + 1)
        for (start in 0 until lastStart) {
            val window = utt.subList(start, minOf(start + n, utt.size))
            if (window.isEmpty()) continue
            val used = HashSet<Int>()
            val sims = ArrayList<Double>(n)
            for (pt in phrase) {
                var bestJ = -1
                var bestV = 0.0
                for ((j, w) in window.withIndex()) {
                    if (j in used) continue
                    val v = jaroWinkler(pt, w)
                    if (v > bestV) { bestV = v; bestJ = j }
                }
                if (bestJ >= 0) used.add(bestJ)
                sims.add(bestV)
            }
            val sim = sims.average()
            if (sim > bestSim) {
                bestSim = sim
                bestUsed = sims.count { it >= 0.8 }
            }
        }
        return bestSim to (bestUsed.toDouble() / utt.size)
    }

    /** Jaro-Winkler similarity in [0,1]; tolerant of short-string ASR mishears. */
    internal fun jaroWinkler(s1: String, s2: String, p: Double = 0.1): Double {
        val j = jaro(s1, s2)
        if (j < 0.7) return j
        var prefix = 0
        val n = minOf(minOf(s1.length, s2.length), 4)
        for (i in 0 until n) {
            if (s1[i] == s2[i]) prefix++ else break
        }
        return j + prefix * p * (1.0 - j)
    }

    private fun jaro(s1: String, s2: String): Double {
        if (s1 == s2) return 1.0
        if (s1.isEmpty() || s2.isEmpty()) return 0.0
        val matchDist = maxOf(maxOf(s1.length, s2.length) / 2 - 1, 0)
        val s1m = BooleanArray(s1.length)
        val s2m = BooleanArray(s2.length)
        var matches = 0
        for (i in s1.indices) {
            val lo = maxOf(0, i - matchDist)
            val hi = minOf(i + matchDist + 1, s2.length)
            for (k in lo until hi) {
                if (!s2m[k] && s2[k] == s1[i]) {
                    s1m[i] = true; s2m[k] = true; matches++; break
                }
            }
        }
        if (matches == 0) return 0.0
        var t = 0.0
        var k = 0
        for (i in s1.indices) {
            if (s1m[i]) {
                while (!s2m[k]) k++
                if (s1[i] != s2[k]) t++
                k++
            }
        }
        t /= 2.0
        val m = matches.toDouble()
        return (m / s1.length + m / s2.length + (m - t) / m) / 3.0
    }
}

/**
 * The command vocabulary as DATA. Adding a command = one row; reordering bugs are gone
 * because coverage scoring (not declaration order) decides the winner.
 *
 * Thresholds: higher = stricter. Short or dangerous commands sit high (EMERGENCY 0.90,
 * PAUSE 0.88, BATTERY 0.86) so ambient half-matches can't fire them. EMERGENCY is precision-
 * leaning for now BY DESIGN: until the countdown-with-abort lands (deferred), a false SOS to a
 * contact is worse than a missed slurred "emergency" the user can repeat. When the countdown
 * ships, lower this for higher recall (the abort window makes a false fire recoverable).
 */
object CommandCatalog {
    val commands: List<CommandSpec> = listOf(
        CommandSpec(NovaCommand.DESCRIBE_FRONT,
            listOf("what's in front", "what is in front", "what's ahead", "what is ahead",
                "what's there", "look ahead", "what's in front of me"),
            Risk.CHEAP, threshold = 0.80),
        CommandSpec(NovaCommand.DISTANCE_QUERY,
            listOf("how far", "how far is it", "how close", "distance"),
            Risk.CHEAP, threshold = 0.82),
        CommandSpec(NovaCommand.FIND_CHAIR,
            listOf("find a chair", "find me a chair", "empty chair", "where can i sit",
                "find chair", "where is a chair", "where's a chair"),
            Risk.NAVIGATIONAL, threshold = 0.80),
        CommandSpec(NovaCommand.FIND_DOOR,
            listOf("find the door", "where's the door", "where is the door", "find door",
                "find the exit", "find exit", "where's the exit"),
            Risk.NAVIGATIONAL, threshold = 0.80),
        CommandSpec(NovaCommand.READ_TEXT,
            listOf("read text", "read this", "read the text", "read it", "read sign",
                "read label", "what does it say", "what does this say"),
            Risk.NAVIGATIONAL, threshold = 0.82),
        CommandSpec(NovaCommand.SCAN_ROOM,
            listOf("scan room", "scan this room", "save this room"),
            Risk.DESTRUCTIVE, threshold = 0.84, requiresConfirm = true),
        CommandSpec(NovaCommand.LOAD_ROOM,
            listOf("load room", "which room"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.EMERGENCY,
            listOf("emergency", "sos", "call for help"),
            Risk.SAFETY, threshold = 0.90, cooldownExempt = true),
        CommandSpec(NovaCommand.CANCEL_EMERGENCY,
            listOf("cancel emergency", "cancel sos", "stop emergency"),
            Risk.SAFETY_CANCEL, threshold = 0.80, cooldownExempt = true),
        CommandSpec(NovaCommand.BATTERY,
            listOf("battery", "battery level", "battery status"),
            Risk.CHEAP, threshold = 0.86),
        CommandSpec(NovaCommand.PAUSE,
            listOf("pause", "pause nova", "stop nova"),
            Risk.NAVIGATIONAL, threshold = 0.88),   // high bar so ambient "stop" can't pause detection
        CommandSpec(NovaCommand.RESUME,
            listOf("resume", "resume nova", "start nova"),
            Risk.NAVIGATIONAL, threshold = 0.85),    // dropped bare "continue" — too ambient
        CommandSpec(NovaCommand.SETTINGS,
            listOf("settings", "open settings", "preferences"),
            Risk.NAVIGATIONAL, threshold = 0.85),
        CommandSpec(NovaCommand.DESCRIBE_ALL,
            listOf("what's around", "what's around me", "describe everything",
                "tell me everything", "around me", "what's nearby", "describe surroundings"),
            Risk.CHEAP, threshold = 0.80),
        CommandSpec(NovaCommand.PEOPLE_NEARBY,
            listOf("people around", "anyone around", "people nearby", "any people",
                "is anyone there", "anyone here", "how many people", "people here"),
            Risk.CHEAP, threshold = 0.82),
        CommandSpec(NovaCommand.REPEAT,
            listOf("say again", "repeat that", "say that again", "what did you say",
                "come again", "repeat"),
            Risk.CHEAP, threshold = 0.85),
        CommandSpec(NovaCommand.TIME_QUERY,
            listOf("what time", "what time is it", "what's the time", "the time",
                "current time", "time is it"),
            Risk.CHEAP, threshold = 0.82),
        CommandSpec(NovaCommand.STATUS,
            listOf("nova status", "system status", "status report", "are you working",
                "server status", "are you connected"),
            Risk.CHEAP, threshold = 0.82),
        CommandSpec(NovaCommand.GO_HOME,
            listOf("go home", "home screen", "main screen", "go to navigation", "navigation screen"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.GO_BACK,
            listOf("go back", "take me back", "previous screen", "close this screen",
                "close settings", "close help"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.SPEAK_SLOWER,
            listOf("speak slower", "talk slower", "slow down speech", "slower speech"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.SPEAK_FASTER,
            listOf("speak faster", "talk faster", "speed up speech", "faster speech"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.HELP_COMMANDS,
            listOf("help", "help commands", "what can i say", "list commands",
                "available commands", "what commands", "show commands"),
            Risk.CHEAP, threshold = 0.86),
        CommandSpec(NovaCommand.OPEN_HELP,
            listOf("open help", "help screen", "show help", "go to help"),
            Risk.NAVIGATIONAL, threshold = 0.84),
        CommandSpec(NovaCommand.OPEN_FINDER,
            listOf("open finder", "find object", "search for object", "object finder",
                "grounding dino", "open search", "find something"),
            Risk.NAVIGATIONAL, threshold = 0.82),
        CommandSpec(NovaCommand.UPDATE_CONTACT,
            listOf("update contact", "change contact", "update emergency",
                "change emergency number", "change my contact", "new contact number"),
            Risk.DESTRUCTIVE, threshold = 0.84, requiresConfirm = true),
    )
}
