from django.urls import reverse, resolve
from django.test import TestCase
# from .views import login

# Create your tests here.
# class LoginTests(TestCase):
# 	def test_login_view_status_code(self):
# 		url = reverse('login')
# 		response = self.client.get(url)
# 		self.assertEquals(response.status_code, 200)

# 	def test_login_url_resolves_login_view(self):
# 		view = resolve('/')
# 		self.assertEquals(view.func, login)

from django.test import TestCase
from boards.models import Patient
from django.utils import timezone
from boards.forms import CreatePatientForm

# models test
class CreatePatientTest(TestCase):

	def create_patient(self, patient_name="only a test to create patient"):
		return Patient.objects.create(patient_name=patient_name, DOB=timezone.now(), sex=1, address_line1="test", address_line2="testl2", address_line3="testl3", occupation="test", marital_status=1, tel_num="314", home_num="159")

	def test_patient_creation(self):
		w = self.create_patient()
		self.assertTrue(isinstance(w, Patient))
		self.assertEqual(w.__str__(), w.patient_name)

	def test_patient_list_view(self):
		w = self.create_patient()
		url = reverse("patient")
		resp = self.client.get(url)

		self.assertEqual(resp.status_code, 200)
		self.assertIn(w.patient_name, resp.content)