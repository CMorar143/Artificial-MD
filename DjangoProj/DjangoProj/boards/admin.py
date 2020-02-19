from django.contrib import admin
from .models import Patient, Family, Visit, Ailment, Reminders, Allergy, Medical_history, Investigation, Medication, Examination #, Doctor 

admin.site.register(Patient)
admin.site.register(Family)
admin.site.register(Visit)
admin.site.register(Ailment)
admin.site.register(Reminders)
admin.site.register(Allergy)
admin.site.register(Medical_history)
admin.site.register(Investigation)
admin.site.register(Medication)
# admin.site.register(Doctor)
admin.site.register(Examination)

# Register your models here.
class ExamAdmin(admin.ModelAdmin):
	fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'gender': admin.VERTICAL}

	# fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'chest_pain': admin.VERTICAL}
	radio_fields = {'hist_diabetes': admin.HORIZONTAL}
	radio_fields = {'hist_heart_disease': admin.HORIZONTAL}
	radio_fields = {'exerc_angina': admin.HORIZONTAL}