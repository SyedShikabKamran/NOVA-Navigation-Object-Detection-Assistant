package com.nova.assistant.util

/**
 * Shared speech utilities.
 *
 * Moved from SetupScreen.kt (H1 dependency): parseSpokenNumber() is now
 * accessible from MainNavigationViewModel for SCAN_ROOM / LOAD_ROOM voice capture.
 */

/**
 * Convert spoken phone number text to a clean digit string.
 * Handles: "zero seven one two..." → "0712..."
 *          "07 1 2 3 4 5 6 7 8 9" → "07123456789"
 *          "+44 7 9 1 2 3 4 5 6 7 8" → "+447912345678"
 */
fun parseSpokenNumber(text: String): String {
    val wordToDigit = mapOf(
        "zero" to "0", "oh" to "0", "o" to "0", "nought" to "0",
        "one" to "1",
        "two" to "2", "to" to "2", "too" to "2",
        "three" to "3",
        "four" to "4", "for" to "4",
        "five" to "5",
        "six" to "6",
        "seven" to "7",
        "eight" to "8", "ate" to "8",
        "nine" to "9", "niner" to "9"
    )
    val lower = text.lowercase().trim()
    val converted = lower
        .replace("+", " + ")
        .split("\\s+".toRegex())
        .joinToString("") { token -> wordToDigit[token] ?: token }
    return if (converted.startsWith("+"))
        "+" + converted.drop(1).filter { it.isDigit() }
    else
        converted.filter { it.isDigit() }
}

/**
 * R19: Resolve a spoken menu index (1-based) from free-form speech.
 * Handles single digits ("one", "1") and double digits ("ten", "eleven", "twelve").
 * Returns null if no number is recognised.
 *
 * Used by LOAD_ROOM audio menu to handle room lists larger than 9 entries.
 */
fun parseSpokenMenuIndex(text: String): Int? {
    val lower = text.lowercase().trim()

    // Check for compound teens and tens first (before single-word matches)
    val compoundMap = mapOf(
        "eleven"    to 11, "twelve"     to 12, "thirteen" to 13,
        "fourteen"  to 14, "fifteen"    to 15, "sixteen"  to 16,
        "seventeen" to 17, "eighteen"   to 18, "nineteen" to 19,
        "ten"       to 10, "twenty"     to 20
    )
    for ((word, num) in compoundMap) {
        if (lower.contains(word)) return num
    }

    // Single-digit words
    val singleMap = mapOf(
        "one" to 1, "1" to 1, "first" to 1,
        "two" to 2, "2" to 2, "second" to 2, "to" to 2, "too" to 2,
        "three" to 3, "3" to 3, "third" to 3,
        "four" to 4, "4" to 4, "fourth" to 4, "for" to 4,
        "five" to 5, "5" to 5, "fifth" to 5,
        "six" to 6, "6" to 6, "sixth" to 6,
        "seven" to 7, "7" to 7, "seventh" to 7,
        "eight" to 8, "8" to 8, "eighth" to 8,
        "nine" to 9, "9" to 9, "ninth" to 9
    )
    for ((word, num) in singleMap) {
        if (lower.contains(word)) return num
    }

    // Fallback: first standalone integer found in text
    return Regex("\\b(\\d+)\\b").find(lower)?.groupValues?.get(1)?.toIntOrNull()
}
