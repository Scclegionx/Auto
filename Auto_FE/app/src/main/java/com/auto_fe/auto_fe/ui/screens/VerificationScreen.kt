package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.service.be.AuthService
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VerificationScreen(
    email: String,
    password: String = "", // Thêm password để autofill
    onVerificationSuccess: (String, String) -> Unit, // Trả về email và password
    onBackClick: () -> Unit
) {
    val authService = remember { AuthService() }
    val coroutineScope = rememberCoroutineScope()
    
    var otp by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var successMessage by remember { mutableStateOf<String?>(null) }
    var canResend by remember { mutableStateOf(true) }
    var resendCountdown by remember { mutableStateOf(0) }

    // Countdown timer for resend
    LaunchedEffect(resendCountdown) {
        if (resendCountdown > 0) {
            kotlinx.coroutines.delay(1000)
            resendCountdown--
        } else {
            canResend = true
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Xác thực Email") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Icon hoặc illustration
            Text(
                text = "✉️",
                style = MaterialTheme.typography.displayLarge
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Hướng dẫn
            Text(
                text = "Nhập mã xác thực",
                style = MaterialTheme.typography.headlineSmall
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = "Chúng tôi đã gửi mã gồm 6 chữ số đến",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
            
            Text(
                text = email,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary,
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // OTP Input Field
            OutlinedTextField(
                value = otp,
                onValueChange = { 
                    if (it.length <= 6 && it.all { char -> char.isDigit() }) {
                        otp = it
                        errorMessage = null
                    }
                },
                label = { Text("Mã OTP (6 số)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                isError = errorMessage != null,
                enabled = !isLoading
            )
            
            if (errorMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = errorMessage!!,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall
                )
            }
            
            if (successMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = successMessage!!,
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.bodySmall
                )
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Verify Button
            Button(
                onClick = {
                    if (otp.length != 6) {
                        errorMessage = "Vui lòng nhập đủ 6 số"
                        return@Button
                    }
                    
                    isLoading = true
                    errorMessage = null
                    
                    coroutineScope.launch {
                        val result = authService.verifyOtp(email, otp)
                        isLoading = false
                        
                        result.onSuccess { response ->
                            successMessage = response.message
                            // Đợi 1 giây rồi chuyển màn hình với email và password
                            kotlinx.coroutines.delay(1000)
                            onVerificationSuccess(email, password)
                        }.onFailure { error ->
                            errorMessage = error.message
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isLoading && otp.length == 6
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text("Xác thực")
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Resend OTP
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "Không nhận được mã? ",
                    style = MaterialTheme.typography.bodyMedium
                )
                
                if (canResend) {
                    TextButton(
                        onClick = {
                            canResend = false
                            resendCountdown = 60
                            successMessage = null
                            errorMessage = null
                            
                            coroutineScope.launch {
                                val result = authService.sendVerificationOtp(email)
                                result.onSuccess { response ->
                                    successMessage = "Đã gửi lại mã OTP"
                                }.onFailure { error ->
                                    errorMessage = error.message
                                    canResend = true
                                    resendCountdown = 0
                                }
                            }
                        }
                    ) {
                        Text("Gửi lại")
                    }
                } else {
                    Text(
                        text = "Gửi lại sau ${resendCountdown}s",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
