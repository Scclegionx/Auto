package com.auto_fe.auto_fe.data.model

data class MedicationLog(
    val id: Long,
    val elderUserId: Long,
    val medicationIds: String,
    val medicationNames: String,
    val medicationCount: Int,
    val reminderTime: String,
    val actualTakenTime: String? = null,
    val status: String, // PENDING, TAKEN, MISSED
    val minutesLate: Int? = null,
    val note: String? = null,
    val fcmSent: Boolean = false,
    val fcmSentTime: String? = null,
    val createdAt: String,
    val updatedAt: String
)

data class MedicationLogHistoryResponse(
    val logs: List<MedicationLog>,
    val statistics: Map<String, Any>
)
