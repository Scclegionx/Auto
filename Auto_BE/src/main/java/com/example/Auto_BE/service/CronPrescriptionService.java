package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.MedicationReminderResponse;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.PrescriptionMapper;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.example.Auto_BE.repository.PrescriptionRepository;
import com.example.Auto_BE.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
public class CronPrescriptionService {
    private final UserRepository userRepository;
    private final PrescriptionRepository prescriptionRepository;
    private final MedicationReminderRepository medicationReminderRepository;
    private final CronSchedulerService cronSchedulerService;

    public CronPrescriptionService(UserRepository userRepository,
                                   PrescriptionRepository prescriptionRepository,
                                   MedicationReminderRepository medicationReminderRepository,
                                   CronSchedulerService cronSchedulerService) {
        this.userRepository = userRepository;
        this.prescriptionRepository = prescriptionRepository;
        this.medicationReminderRepository = medicationReminderRepository;
        this.cronSchedulerService = cronSchedulerService;
    }

    @Transactional
    public BaseResponse<PrescriptionResponse> create(PrescriptionCreateRequest prescriptionCreateRequest,
                                                     Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

//        // 2. Validate medication reminders data
//        validateMedicationReminders(prescriptionCreateRequest.getMedicationReminders());

        Prescriptions prescription = PrescriptionMapper.toEntity(prescriptionCreateRequest, user);

        if (prescriptionCreateRequest.getMedicationReminders() != null &&
                !prescriptionCreateRequest.getMedicationReminders().isEmpty()) {

            List<MedicationReminder> medicationReminders = prescriptionCreateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, user, prescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());

            prescription.setMedicationReminders(medicationReminders);
        }

        Prescriptions savedPrescription = prescriptionRepository.save(prescription);

        // Tạo cron jobs cho medication reminders
        if (savedPrescription.getMedicationReminders() != null &&
                !savedPrescription.getMedicationReminders().isEmpty()) {

            for (MedicationReminder reminder : savedPrescription.getMedicationReminders()) {
                try {
                    cronSchedulerService.scheduleWithCron(reminder.getId());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (savedPrescription.getMedicationReminders() == null) ? List.of() :
                        savedPrescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(savedPrescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_CREATED)
                .data(response)
                .build();
    }

    @Transactional
    public BaseResponse<PrescriptionResponse> update(Long prescriptionId,
                                                     PrescriptionCreateRequest prescriptionUpdateRequest,
                                                     Authentication authentication) {
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find existing prescription
        Prescriptions existingPrescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!existingPrescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

//        // Validate new data
//        validateMedicationReminders(prescriptionUpdateRequest.getMedicationReminders());

        // huy tất cả cron jobs cũ
        List<MedicationReminder> oldReminders = existingPrescription.getMedicationReminders();
        if (oldReminders != null && !oldReminders.isEmpty()) {
            for (MedicationReminder oldReminder : oldReminders) {
                try {
                    cronSchedulerService.cancelCronSchedule(oldReminder.getId());
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }

        //cập nhật thông tin prescription
        existingPrescription.setName(prescriptionUpdateRequest.getName());
        existingPrescription.setDescription(prescriptionUpdateRequest.getDescription());
        existingPrescription.setImageUrl(prescriptionUpdateRequest.getImageUrl());

        // tao mới medication reminders nếu có
        if (prescriptionUpdateRequest.getMedicationReminders() != null &&
                !prescriptionUpdateRequest.getMedicationReminders().isEmpty()) {

            // Delete old reminders
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }

            // Create new reminders
            List<MedicationReminder> newReminders = prescriptionUpdateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, user, existingPrescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());

            existingPrescription.setMedicationReminders(newReminders);
        } else {
            // No new reminders
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }
            existingPrescription.setMedicationReminders(null);
        }

        Prescriptions updatedPrescription = prescriptionRepository.save(existingPrescription);

        // tạo cron jobs mới cho medication reminders
        if (updatedPrescription.getMedicationReminders() != null &&
                !updatedPrescription.getMedicationReminders().isEmpty()) {

            for (MedicationReminder newReminder : updatedPrescription.getMedicationReminders()) {
                try {
                    cronSchedulerService.scheduleWithCron(newReminder.getId());
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (updatedPrescription.getMedicationReminders() == null) ? List.of() :
                        updatedPrescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(updatedPrescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_UPDATED)
                .data(response)
                .build();
    }

    public BaseResponse<PrescriptionResponse> getById(Long prescriptionId, Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (prescription.getMedicationReminders() == null) ? List.of() :
                        prescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(prescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(response)
                .build();
    }

    public BaseResponse<List<PrescriptionResponse>> getAllByUser(Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        List<Prescriptions> prescriptions = prescriptionRepository.findByUserIdAndIsActiveTrue(user.getId());

        List<PrescriptionResponse> prescriptionResponses = prescriptions.stream()
                .map(prescription -> {
                    List<MedicationReminderResponse> medicationReminderResponses =
                            (prescription.getMedicationReminders() == null) ? List.of() :
                                    prescription.getMedicationReminders().stream()
                                            .map(PrescriptionMapper::toResponse)
                                            .toList();

                    return PrescriptionMapper.toResponse(prescription, medicationReminderResponses);
                })
                .toList();

        return BaseResponse.<List<PrescriptionResponse>>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(prescriptionResponses)
                .build();
    }

    @Transactional
    public BaseResponse<String> delete(Long prescriptionId, Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        // hủy tất cả cron jobs liên quan đến medication reminders
        if (prescription.getMedicationReminders() != null && !prescription.getMedicationReminders().isEmpty()) {
            for (MedicationReminder reminder : prescription.getMedicationReminders()) {
                try {
                    cronSchedulerService.cancelCronSchedule(reminder.getId());
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }

        prescriptionRepository.delete(prescription);

        return BaseResponse.<String>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_DELETED)
                .data(null)
                .build();
    }

//    private void validateMedicationReminders(List<?> medicationReminders) {
//
//    }
}