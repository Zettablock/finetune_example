TEMPLATE_PROMPT = """You are a helpful medical chatbot. Below is a user's message. Write a response that appropriately completes the request. 

### Message:
{}

### Response:
{}
"""

# TEMPLATE_PROMPT = """Below is an email. You must classify whether or not this email is spam. If the email is indeed spam, print "spam". If the email is not spam, print "ham". 

# ### Subject:
# {}

# ### Message:
# {}

# ### Label:
# {}"""

# FEWSHOT_PROMPT = """Here are a few examples. 

# ### Subject:
# Unbelievable Discount on Designer Watches! Just $19.99!

# ### Message:
# Unlock the secret to luxury at a fraction of the price! Our exclusive collection of designer watches is now available for only $19.99! 
# Don’t miss out on this once-in-a-lifetime offer. Click now to secure your stylish timepiece! #BestDeals #LimitedOffer #DesignerWatches. 
# Quality that exceeds your expectations. Hurry, this offer won’t last long!

# ### Label:
# spam


# ### Subject:
# Team Meeting Reminder - August 3rd at 10:00 AM

# ### Message:
# Hi Team,

# Just a reminder that we have a project update meeting tomorrow at 10 AM in the conference room.

# See you there!

# Best,
# Aidan

# ### Label:
# ham

# """