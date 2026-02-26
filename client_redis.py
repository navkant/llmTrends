"""Basic connection example.
"""

import redis

r = redis.Redis(
    host='redis-18565.crce217.ap-south-1-1.ec2.cloud.redislabs.com',
    port=18565,
    decode_responses=True,
    username="default",
    password="VtlCt9N4NKJSqqJH7F0I00EPkkFDL6Jp",
)

success = r.set('foo', 'bar')
# True

result = r.get('foo')
print(result)
# >>> bar

