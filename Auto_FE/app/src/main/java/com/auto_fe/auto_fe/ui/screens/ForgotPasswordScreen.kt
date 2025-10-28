package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.ForgotPasswordService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ForgotPasswordScreen(
    onBackClick: () -> Unit = {},
    onSuccessNavigateToLogin: () -> Unit = {}
) {
    val forgotPasswordService = remember { ForgotPasswordService() }
    val coroutineScope = rememberCoroutineScope()
    
    var email by remember { mutableStateOf("") }
    var otp by remember { mutableStateOf("") }
    var currentStep by remember { mutableStateOf(1) } // 1: Email, 2: OTP, 3: Success
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var successMessage by remember { mutableStateOf<String?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        when(currentStep) {
                            1 -> "Quên mật khẩu"
                            2 -> "Xác thực OTP"
                            else -> "Hoàn tất"
                        }
                    ) 
                },
                navigationIcon = {
                    IconButton(onClick = {
                        if (currentStep == 2) {
                            // Quay lại bước nhập email
                            currentStep = 1
                            otp = ""
                            errorMessage = null
                        } else {
                            onBackClick()
                        }
                    }) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface,
                    navigationIconContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(40.dp))
            
            // Icon
            Surface(
                modifier = Modifier.size(100.dp),
                shape = RoundedCornerShape(50.dp),
                color = DarkPrimary.copy(alpha = 0.2f)
            ) {
                Box(
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Email,
                        contentDescription = "Email",
                        tint = DarkPrimary,
                        modifier = Modifier.size(50.dp)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Title
            Text(
                text = when(currentStep) {
                    1 -> "Khôi phục mật khẩu"
                    2 -> "Nhập mã xác thực"
                    else -> "Thành công!"
                },
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Description
            Text(
                text = when(currentStep) {
                    1 -> "Nhập email của bạn để nhận mã OTP xác thực."
                    2 -> "Mã OTP đã được gửi đến email: $email"
                    else -> "Mật khẩu mới đã được gửi đến email của bạn."
                },
                fontSize = 14.sp,
                color = DarkOnSurface.copy(alpha = 0.7f),
                textAlign = TextAlign.Center,
                lineHeight = 20.sp
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // Step 1: Email input
            if (currentStep == 1) {
                OutlinedTextField(
                    value = email,
                    onValueChange = { 
                        email = it
                        errorMessage = null
                    },
                    label = { Text("Email") },
                    leadingIcon = {
                        Icon(Icons.Default.Email, "Email", tint = DarkPrimary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    ),
                    isError = errorMessage != null
                )
            }
            
            // Step 2: OTP input
            if (currentStep == 2) {
                OutlinedTextField(
                    value = otp,
                    onValueChange = { 
                        if (it.length <= 6 && it.all { char -> char.isDigit() }) {
                            otp = it
                            errorMessage = null
                        }
                    },
                    label = { Text("Mã OTP (6 chữ số)") },
                    leadingIcon = {
                        Icon(Icons.Default.Lock, "OTP", tint = DarkPrimary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    ),
                    isError = errorMessage != null,
                    placeholder = { Text("000000") }
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Resend OTP button
                TextButton(
                    onClick = {
                        coroutineScope.launch {
                            isLoading = true
                            errorMessage = null
                            
                            val result = forgotPasswordService.sendForgotPasswordOtp(email)
                            
                            isLoading = false
                            
                            result.onSuccess { response ->
                                successMessage = "Đã gửi lại mã OTP"
                                kotlinx.coroutines.delay(2000)
                                successMessage = null
                            }.onFailure { error ->
                                errorMessage = error.message
                            }
                        }
                    },
                    enabled = !isLoading
                ) {
                    Text(
                        "Gửi lại mã OTP",
                        color = DarkPrimary,
                        fontSize = 14.sp
                    )
                }
            }
            
            // Error message
            if (errorMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = errorMessage!!,
                    color = DarkError,
                    fontSize = 12.sp,
                    modifier = Modifier.fillMaxWidth()
                )
            }
            
            if (successMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = DarkPrimary.copy(alpha = 0.2f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "✓ $successMessage",
                        color = DarkPrimary,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(16.dp),
                        lineHeight = 20.sp
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Submit button
            Button(
                onClick = {
                    when (currentStep) {
                        1 -> {
                            // Bước 1: Gửi OTP
                            if (email.isBlank()) {
                                errorMessage = "Vui lòng nhập email"
                                return@Button
                            }
                            
                            if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
                                errorMessage = "Email không hợp lệ"
                                return@Button
                            }
                            
                            coroutineScope.launch {
                                isLoading = true
                                errorMessage = null
                                
                                val result = forgotPasswordService.sendForgotPasswordOtp(email)
                                
                                isLoading = false
                                
                                result.onSuccess { response ->
                                    successMessage = response.message
                                    kotlinx.coroutines.delay(1000)
                                    currentStep = 2
                                    successMessage = null
                                }.onFailure { error ->
                                    errorMessage = error.message
                                }
                            }
                        }
                        2 -> {
                            // Bước 2: Verify OTP
                            if (otp.length != 6) {
                                errorMessage = "Vui lòng nhập đủ 6 số"
                                return@Button
                            }
                            
                            coroutineScope.launch {
                                isLoading = true
                                errorMessage = null
                                
                                val verifyResult = forgotPasswordService.verifyForgotPasswordOtp(email, otp)
                                
                                verifyResult.onSuccess {
                                    // Verify thành công, gọi API reset password
                                    val resetResult = forgotPasswordService.forgotPassword(email)
                                    
                                    isLoading = false
                                    
                                    resetResult.onSuccess { response ->
                                        successMessage = response.message
                                        currentStep = 3
                                        
                                        // Chờ 3 giây rồi chuyển về màn hình login
                                        kotlinx.coroutines.delay(3000)
                                        onSuccessNavigateToLogin()
                                    }.onFailure { error ->
                                        errorMessage = error.message
                                    }
                                }.onFailure { error ->
                                    isLoading = false
                                    errorMessage = error.message
                                }
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = !isLoading && currentStep < 3,
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkPrimary,
                    disabledContainerColor = DarkPrimary.copy(alpha = 0.5f)
                )
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        color = DarkOnPrimary,
                        modifier = Modifier.size(24.dp)
                    )
                } else {
                    Text(
                        when(currentStep) {
                            1 -> "Gửi mã OTP"
                            2 -> "Xác nhận"
                            else -> "Hoàn tất"
                        },
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Back to login (chỉ hiển thị ở bước 1 hoặc khi hoàn tất)
            if (currentStep == 1 || currentStep == 3) {
                TextButton(
                    onClick = onBackClick,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "Quay lại đăng nhập",
                        color = DarkPrimary,
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}
