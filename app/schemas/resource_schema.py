from marshmallow import Schema, validate, fields, validates, ValidationError
from pprint import pprint
class TodoSchema(Schema):
    name = fields.String(required=True, error_messages={'required': "name is required"})
    email = fields.String(required= True, data_key="emailAddress", error_messages={'required': 'email is required'})

    @validates("email")
    def validate_names(self, value: str, data_key: str) -> None:
        if len(value) < 3:
            raise ValidationError("Invalid email")


s = TodoSchema()


data = {"name": "Mike", "email": "foo@bar.com"}
result = s.dump(data) # does not call the custom message

pprint(result)
data = {"name": "Mike", "emailAddress": "foo@bar.com"}
result = s.load(data)  #call the custom message


pprint(result)
