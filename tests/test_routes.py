# tests/test_routes.py
import unittest
from app import app

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<html', response.data.lower())

    def test_compare_route_no_files(self):
        response = self.app.post('/compare', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Both files are required.', response.data)

    def test_send_text_similarity(self):
        response = self.app.post('/send', data={
            'text1': 'Artificial intelligence is evolving.',
            'text2': 'Artificial intelligence is growing rapidly.'
        })
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('similarity', json_data)
        self.assertIn('message', json_data)

    def test_check_plagiarism_empty(self):
        response = self.app.post('/check_plagiarism', data={})
        self.assertEqual(response.status_code, 400)
