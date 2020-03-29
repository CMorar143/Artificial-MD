from django.contrib import admin
from .models import Patient, Patient_Ailment, Patient_Allergy, Patient_Family, Family, Visit, Ailment, Reminder, Allergy, Medical_history, Investigation, Medication, Patient_Medication, Examination #, Ailment_Medication, Allergy_Medication

admin.site.register(Patient)
admin.site.register(Patient_Ailment)
admin.site.register(Patient_Allergy)
admin.site.register(Patient_Family)
admin.site.register(Family)
admin.site.register(Visit)
admin.site.register(Ailment)
# admin.site.register(Ailment_Medication)
admin.site.register(Reminder)
admin.site.register(Allergy)
# admin.site.register(Allergy_Medication)
admin.site.register(Medical_history)
admin.site.register(Investigation)
admin.site.register(Medication)
admin.site.register(Patient_Medication)
admin.site.register(Examination)


# Register your models here.
class ExamAdmin(admin.ModelAdmin):
	fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'gender': admin.VERTICAL}

	radio_fields = {'chest_pain': admin.VERTICAL}
	radio_fields = {'hist_diabetes': admin.HORIZONTAL}
	radio_fields = {'hist_heart_disease': admin.HORIZONTAL}
	radio_fields = {'exerc_angina': admin.HORIZONTAL}