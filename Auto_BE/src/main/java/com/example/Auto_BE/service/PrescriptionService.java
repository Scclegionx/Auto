//package com.example.Auto_BE.service;
//
//import com.example.Auto_BE.dto.BaseResponse;
//import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
//import com.example.Auto_BE.dto.response.MedicationReminderResponse;
//import com.example.Auto_BE.dto.response.PrescriptionResponse;
//import com.example.Auto_BE.entity.MedicationReminder;
//import com.example.Auto_BE.entity.Prescriptions;
//import com.example.Auto_BE.entity.User;
//import com.example.Auto_BE.exception.BaseException;
//import com.example.Auto_BE.mapper.PrescriptionMapper;
//import com.example.Auto_BE.repository.MedicationReminderRepository;
//import com.example.Auto_BE.repository.PrescriptionRepository;
//import com.example.Auto_BE.repository.UserRepository;
//import jakarta.transaction.Transactional;
//import org.springframework.security.core.Authentication;
//import org.springframework.stereotype.Service;
//
//import java.util.List;
//import java.util.stream.Collectors;
//
//import static com.example.Auto_BE.constants.ErrorMessages.USER_NOT_FOUND;
//import static com.example.Auto_BE.constants.SuccessMessage.PRESCRIPTION_CREATED;
//import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;
//
//@Service
//public class PrescriptionService {
//    private final UserRepository userRepository;
//    private final PrescriptionRepository prescriptionRepository;
//    private final MedicationReminderRepository medicationReminderRepository;
//
//    public PrescriptionService(UserRepository userRepository,
//                               PrescriptionRepository prescriptionRepository,
//                               MedicationReminderRepository medicationReminderRepository) {
//        this.userRepository = userRepository;
//        this.prescriptionRepository = prescriptionRepository;
//        this.medicationReminderRepository = medicationReminderRepository;
//    }
//
//    @Transactional
//    public BaseResponse<PrescriptionResponse> create(PrescriptionCreateRequest prescriptionCreateRequest,
//                                                     Authentication authentication) {
//        // Kiểm tra người dùng có tồn tại không
//        User user = userRepository.findByEmail(authentication.getName())
//                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
//
//        // Validate dữ liệu medication reminders trước khi tạo entity
//        if (prescriptionCreateRequest.getMedicationReminders() != null) {
//            for (var reminder : prescriptionCreateRequest.getMedicationReminders()) {
//                if (reminder.getType() == null) {
//                    throw new BaseException.BadRequestException("Loại thuốc không được để trống");
//                }
//                if (reminder.getName() == null || reminder.getName().trim().isEmpty()) {
//                    throw new BaseException.BadRequestException("Tên thuốc không được để trống");
//                }
//                if (reminder.getDaysOfWeek() == null || reminder.getDaysOfWeek().trim().isEmpty()) {
//                    throw new BaseException.BadRequestException("Ngày trong tuần không được để trống");
//                }
//                if (!reminder.getDaysOfWeek().matches("^[01]{7}$")) {
//                    throw new BaseException.BadRequestException("Ngày trong tuần phải là chuỗi 7 ký tự 0/1");
//                }
//                if (reminder.getReminderTimes() == null || reminder.getReminderTimes().isEmpty()) {
//                    throw new BaseException.BadRequestException("Cần ít nhất 1 giờ nhắc");
//                }
//            }
//        }
//
//        Prescriptions prescription = PrescriptionMapper.toEntity(prescriptionCreateRequest, user);
//
//        if(prescriptionCreateRequest.getMedicationReminders()!=null && !prescriptionCreateRequest.getMedicationReminders().isEmpty()) {
//            List<MedicationReminder> mrs = prescriptionCreateRequest.getMedicationReminders().stream()
//                    .map(mr -> PrescriptionMapper.toEntities(mr, user, prescription))
//                    .flatMap(List::stream)
//                    .collect(Collectors.toList());
//            prescription.setMedicationReminders(mrs);
//        }
//
//        Prescriptions savedPrescription = prescriptionRepository.save(prescription);
//
//        List<MedicationReminderResponse> medicationReminderResponses =(savedPrescription.getMedicationReminders()==null)? List.of():
//                savedPrescription.getMedicationReminders().stream()
//                        .map(PrescriptionMapper::toResponse)
//                        .toList();
//
//        PrescriptionResponse response = PrescriptionMapper.toResponse(savedPrescription, medicationReminderResponses);
//        return BaseResponse.<PrescriptionResponse>builder()
//                .status(SUCCESS)
//                .message(PRESCRIPTION_CREATED)
//                .data(response)
//                .build();
//    }
//}
