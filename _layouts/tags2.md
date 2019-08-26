---
title: "Notes & Resources"
permalink: /tags2/
layout: single
---


{% for tag in site.tags %}
	<span> {{ tag }} </span>
	{% for post in tag %}
	  <div class="post_info">
	    <li>
	    	 <span>[   {{ post.date | date:"%Y-%m-%d" }}   ]</span>
	         <a href="{{ post.url }}">{{ post.title }}</a>
	    </li>
	    </div>
	  {% endfor %}
{% endfor %}

