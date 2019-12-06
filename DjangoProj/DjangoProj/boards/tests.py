from django.urls import reverse, resolve
from django.test import TestCase
from .views import login

# Create your tests here.
class LoginTests(TestCase):
	def test_login_view_status_code(self):
		url = reverse('login')
		response = self.client.get(url)
		self.assertEquals(response.status_code, 200)

	def test_login_url_resolves_login_view(self):
		view = resolve('/')
		self.assertEquals(view.func, login)