from django.contrib import admin
from .models import Patient, Patient_Ailment, Patient_Allergy, Family, Visit, Ailment, Ailment_Medication, Reminder, Allergy, Allergy_Medication, Medical_history, Investigation, Medication, Examination

admin.site.register(Patient)
admin.site.register(Patient_Ailment)
admin.site.register(Patient_Allergy)
admin.site.register(Family)
admin.site.register(Visit)
admin.site.register(Ailment)
admin.site.register(Ailment_Medication)
admin.site.register(Reminder)
admin.site.register(Allergy)
admin.site.register(Allergy_Medication)
admin.site.register(Medical_history)
admin.site.register(Investigation)
admin.site.register(Medication)
admin.site.register(Examination)

# Register your models here.
class ExamAdmin(admin.ModelAdmin):
	fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'gender': admin.VERTICAL}

	radio_fields = {'chest_pain': admin.VERTICAL}
	radio_fields = {'hist_diabetes': admin.HORIZONTAL}
	radio_fields = {'hist_heart_disease': admin.HORIZONTAL}
	radio_fields = {'exerc_angina': admin.HORIZONTAL}