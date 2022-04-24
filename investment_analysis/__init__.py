__version__ = '0.0.0'
import requests_cache

requests_cache.install_cache(cache_name="/tmp/.requests_cache", backend='sqlite')
