package com.auto_fe.auto_fe.base

class ConfirmationRequirement(
    val originalInput: String,
    val confirmationQuestion: String,
    val onConfirmed: suspend () -> String,
    val isMultipleContacts: Boolean = false,
    val actionType: String = "",
    val actionData: String = ""
) : Exception("Confirmation required")

