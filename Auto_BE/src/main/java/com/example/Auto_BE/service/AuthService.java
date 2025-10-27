package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.ForgotPasswordRequest;
import com.example.Auto_BE.dto.request.LoginRequest;
import com.example.Auto_BE.dto.request.RegisterRequest;
import com.example.Auto_BE.dto.request.ResendVerificationRequest;
import com.example.Auto_BE.dto.request.SendVerificationRequest;
import com.example.Auto_BE.dto.request.VerifyOtpRequest;
import com.example.Auto_BE.dto.response.LoginResponse;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.Verification;
import com.example.Auto_BE.event.ForgotPasswordEvent;
import com.example.Auto_BE.event.UserRegistrationEvent;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.repository.VerificationRepository;
import com.example.Auto_BE.utils.JwtUtils;
import com.example.Auto_BE.utils.RandomPassword;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.Random;

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {
    private final JwtUtils jwtUtils;
    private final UserRepository userRepository;
    private final AuthenticationManager authenticationManager;
    private final PasswordEncoder passwordEncoder;
    private final ApplicationEventPublisher applicationEventPublisher;
    private final VerificationRepository verificationRepository;

    public BaseResponse<LoginResponse> login(LoginRequest loginRequest) {
        User user = userRepository.findByEmail(loginRequest.getEmail())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        if (!user.getIsActive()) {
            throw new BaseException.BadRequestException(UNVERIFIED_ACCOUNT);
        }
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        loginRequest.getEmail(),
                        loginRequest.getPassword()
                )
        );
        String accessToken = jwtUtils.generateAccessToken(authentication);
        
        // Tạo UserInfo để trả về
        LoginResponse.UserInfo userInfo = LoginResponse.UserInfo.builder()
                .id(user.getId())
                .email(user.getEmail())
                .name(user.getFullName())
                .build();
        
        LoginResponse loginResponse = LoginResponse.builder()
                .accessToken(accessToken)
                .user(userInfo)
                .build();
        
        return BaseResponse.<LoginResponse>builder()
                .status(SUCCESS)
                .message(USER_LOGGED_IN)
                .data(loginResponse)
                .build();
    }

    @Transactional
    public BaseResponse<Void> register(RegisterRequest registerRequest) {
        Optional<User> existingUserOpt = userRepository.findByEmail(registerRequest.getEmail());
        if (existingUserOpt.isPresent()) {
            User existingUser = existingUserOpt.get();
            if (!existingUser.getIsActive()) {
                // Nếu email đã tồn tại nhưng chưa active, gửi lại mã OTP mới
                existingUser.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
                userRepository.save(existingUser);

                // Xóa các OTP cũ
                verificationRepository.deleteAllByUser(existingUser);
                
                // Tạo và gửi OTP mới
                String otp = generateOtp();
                
                Verification verification = new Verification();
                verification.setUser(existingUser)
                        .setToken(otp)
                        .setExpiredAt(LocalDateTime.now().plusMinutes(5));
                verificationRepository.save(verification);

                applicationEventPublisher.publishEvent(new UserRegistrationEvent(this, verification));

                return BaseResponse.<Void>builder()
                        .status(SUCCESS)
                        .message(EMAIL_RESENT)
                        .build();
            } else {
                // Đã active thì báo lỗi
                throw new BaseException.ConflictException(EMAIL_ALREADY_EXISTS);
            }
        }

        User newUser = new User();
        newUser.setEmail(registerRequest.getEmail())
                .setPassword(passwordEncoder.encode(registerRequest.getPassword()))
                .setIsActive(false); // Chưa active, phải verify email trước
        userRepository.save(newUser);

        // Tự động gửi mã OTP sau khi đăng ký
        String otp = generateOtp();
        
        Verification verification = new Verification();
        verification.setUser(newUser)
                .setToken(otp)
                .setExpiredAt(LocalDateTime.now().plusMinutes(5));
        verificationRepository.save(verification);

        // Gửi email chứa mã OTP
        applicationEventPublisher.publishEvent(new UserRegistrationEvent(this, verification));
        
        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(USER_REGISTERED)
                .build();
    }

    /**
     * Gửi mã OTP xác thực email (user chủ động gọi từ FE)
     */
    @Transactional
    public BaseResponse<Void> sendVerificationOtp(SendVerificationRequest request) {
        User user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        if (user.getIsActive()) {
            throw new BaseException.BadRequestException(USER_ALREADY_VERIFIED);
        }

        // Xóa các OTP cũ của user (nếu có)
        verificationRepository.deleteAllByUser(user);

        // Tạo mã OTP 6 số
        String otp = generateOtp();
        
        // Lưu OTP vào database, hết hạn sau 5 phút
        Verification verification = new Verification();
        verification.setUser(user)
                .setToken(otp)
                .setExpiredAt(LocalDateTime.now().plusMinutes(5));
        verificationRepository.save(verification);

        // Gửi email chứa mã OTP
        applicationEventPublisher.publishEvent(new UserRegistrationEvent(this, verification));

        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(VERIFICATION_EMAIL_SENT)
                .build();
    }

    /**
     * Xác thực mã OTP
     */
    @Transactional
    public BaseResponse<Void> verifyOtp(VerifyOtpRequest request) {
        User user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        if (user.getIsActive()) {
            throw new BaseException.BadRequestException(USER_ALREADY_VERIFIED);
        }

        // Tìm verification theo user và OTP
        Verification verification = verificationRepository.findByUserAndToken(user, request.getOtp())
                .orElseThrow(() -> new BaseException.BadRequestException("Mã OTP không đúng"));

        // Kiểm tra OTP đã hết hạn chưa
        if (verification.getExpiredAt().isBefore(LocalDateTime.now())) {
            verificationRepository.delete(verification);
            throw new BaseException.BadRequestException("Mã OTP đã hết hạn");
        }

        // Active user
        user.setIsActive(true);
        userRepository.save(user);

        // Xóa tất cả verification của user
        verificationRepository.deleteAllByUser(user);

        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(EMAIL_VERIFIED)
                .build();
    }

    /**
     * Tạo mã OTP 6 số ngẫu nhiên
     */
    private String generateOtp() {
        Random random = new Random();
        int otp = 100000 + random.nextInt(900000); // Tạo số từ 100000 đến 999999
        return String.valueOf(otp);
    }

    @Transactional
    public BaseResponse<Void> verifyEmail(String token) {
        Verification verification = verificationRepository.findByToken(token)
                .orElseThrow(() -> new BaseException.BadRequestException(BAD_REQUEST));

        if (verification.getExpiredAt().isBefore(java.time.LocalDateTime.now())&&
                jwtUtils.validateVerificationToken(token)) {
            throw new BaseException.BadRequestException(BAD_REQUEST);
        }
        if (verification.getUser().getIsActive()) {
            throw new BaseException.BadRequestException(USER_ALREADY_VERIFIED);
        }

        User user = verification.getUser();
        user.setIsActive(true);
        userRepository.save(user);

        // Xóa tất cả verification token cũ của user (bao gồm token hiện tại)
        verificationRepository.deleteAllByUser(user);
        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(EMAIL_VERIFIED)
                .build();
    }

    public BaseResponse<Void> resendVerificationEmail(ResendVerificationRequest resendVerificationRequest) {
        User user = userRepository.findByEmail(resendVerificationRequest.getEmail())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        if (user.getIsActive()) {
            throw new BaseException.BadRequestException(USER_ALREADY_VERIFIED);
        }

        String verificationToken = jwtUtils.generateVerificationToken(user.getEmail());

        Verification verification = new Verification();
        verification.setUser(user)
                .setToken(verificationToken)
                .setExpiredAt(jwtUtils.getVerificationTokenExpirationTime(verificationToken));
        verificationRepository.save(verification);

        applicationEventPublisher.publishEvent(new UserRegistrationEvent(this, verification));

        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(VERIFICATION_EMAIL_SENT)
                .build();
    }

    public BaseResponse<Void> forgotPassword(ForgotPasswordRequest forgotPasswordRequest) {
        User user = userRepository.findByEmail(forgotPasswordRequest.getEmail())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        String newPassword = RandomPassword.generateRandomPassword();
        user.setPassword(passwordEncoder.encode(newPassword));
        userRepository.save(user);
        applicationEventPublisher.publishEvent(new ForgotPasswordEvent(this, user.getEmail(), newPassword));
        return BaseResponse.<Void>builder()
                .status(SUCCESS)
                .message(PASSWORD_RESET)
                .build();
    }

}