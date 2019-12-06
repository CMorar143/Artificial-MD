from django.urls import reverse
from django.test import TestCase

# Create your tests here.
class LoginTests(TestCase):
	def test_login_view_status_code(self):
		url = reverse('login')
		response = self.client.get(url)
		self.assertEquals(response.status_code, 200)