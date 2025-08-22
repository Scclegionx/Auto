package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.DeviceTokenRequest;
import com.example.Auto_BE.dto.response.DeviceTokenResponse;
import com.example.Auto_BE.entity.DeviceToken;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.DeviceTokenMapper;
import com.example.Auto_BE.repository.DeviceTokenRepository;
import com.example.Auto_BE.repository.UserRepository;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import static com.example.Auto_BE.constants.ErrorMessages.USER_NOT_FOUND;
import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;
import static com.example.Auto_BE.constants.SuccessMessage.TOKEN_REGISTERED;

@Service
public class DeviceTokenService {
    private final DeviceTokenRepository deviceTokenRepository;
    private final UserRepository userRepository;

    public DeviceTokenService(DeviceTokenRepository deviceTokenRepository, UserRepository userRepository) {
        this.deviceTokenRepository = deviceTokenRepository;
        this.userRepository = userRepository;
    }

    public BaseResponse<DeviceTokenResponse> registerDeviceToken(DeviceTokenRequest deviceTokenRequest, Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
//        // Kiểm tra xem token đã tồn tại chưa
//        if (deviceTokenRepository.existsByUserIdAndToken(user.getId(), deviceTokenRequest.getToken())) {
//            throw new BaseException.BadRequestException("Token thiết bị đã được đăng ký trước đó");
//        }

        DeviceToken deviceToken = DeviceTokenMapper.toEntity(deviceTokenRequest, user);
        deviceTokenRepository.save(deviceToken);

        DeviceTokenResponse deviceTokenResponse = DeviceTokenMapper.toResponse(deviceToken);

        return BaseResponse.<DeviceTokenResponse>builder()
                .status(SUCCESS)
                .message(TOKEN_REGISTERED)
                .data(deviceTokenResponse)
                .build();
    }

}