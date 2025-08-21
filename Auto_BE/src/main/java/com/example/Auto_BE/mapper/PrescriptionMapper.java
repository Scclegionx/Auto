package com.example.Auto_BE.mapper;


import com.example.Auto_BE.dto.request.MedicationReminderCreateRequest;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.MedicationReminderResponse;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;
import com.example.Auto_BE.entity.User;

import java.util.ArrayList;
import java.util.List;

public class PrescriptionMapper {

    public static Prescriptions toEntity(PrescriptionCreateRequest dto, User user) {
        Prescriptions p = new Prescriptions()
                .setName(dto.getName())
                .setDescription(dto.getDescription())
                .setImageUrl(dto.getImageUrl())
                .setIsActive(true)
                .setUser(user);
        return p;
    }

    // Mapper snippet (đặt trong PrescriptionMapper hoặc class tiện ích của bạn)
    public static List<MedicationReminder> toEntities(
            MedicationReminderCreateRequest dto,
            User user,
            Prescriptions p
    ) {
        if (dto.getReminderTimes() == null || dto.getReminderTimes().isEmpty()) {
            return List.of();
        }

        // loại null/blank, unique & sort
        List<String> times = dto.getReminderTimes().stream()
                .filter(t -> t != null && !t.isBlank())
                .map(String::trim)
                .distinct()
                .sorted()
                .toList();
        List<MedicationReminder> list = new ArrayList<>(times.size());
        for (String timeStr : times) {
            MedicationReminder mr = new MedicationReminder()
                    .setName(dto.getName())
                    .setDescription(dto.getDescription())
                    .setType(dto.getType())
                    .setReminderTime(timeStr)              // mỗi bản ghi 1 giờ
                    .setDaysOfWeek(dto.getDaysOfWeek())
                    .setIsActive(true)
                    .setUser(user)
                    .setPrescription(p);
            list.add(mr);
        }
        return list;
    }


    public static PrescriptionResponse toResponse(Prescriptions p, List<MedicationReminderResponse> reminders) {
        return PrescriptionResponse.builder()
                .id(p.getId())
                .name(p.getName())
                .description(p.getDescription())
                .imageUrl(p.getImageUrl())
                .isActive(p.getIsActive())
                .userId(p.getUser() != null ? p.getUser().getId() : null)
                .medicationReminders(reminders)
                .build();
    }

    public static MedicationReminderResponse toResponse(MedicationReminder mr) {
        return MedicationReminderResponse.builder()
                .id(mr.getId())
                .name(mr.getName())
                .description(mr.getDescription())
                .type(mr.getType())
                .reminderTime(mr.getReminderTime())
                .daysOfWeek(mr.getDaysOfWeek())
                .isActive(mr.getIsActive())
                .prescriptionId(mr.getPrescription() != null ? mr.getPrescription().getId() : null)
                .userId(mr.getUser() != null ? mr.getUser().getId() : null)
                .build();
    }
}
