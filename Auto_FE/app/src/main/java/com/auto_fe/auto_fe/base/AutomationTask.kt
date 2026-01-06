package com.auto_fe.auto_fe.base

interface AutomationTask {
    suspend fun execute(input: String): String
}

