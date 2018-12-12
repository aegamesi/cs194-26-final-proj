$(document).ready(function () {
    $('img').each(function () {
        var img = $(this);
        var a = $(this).wrap('<a></a>').parent();
        a.attr('href', img.attr('src'));
        a.featherlight('image');
        img.attr('title', 'Click to enlarge image');
    });

    $('script').each(function () {
        var type = $(this).attr('type')
        if (type !== 'latex' && type !== 'latex-inline') {
            return;
        }
        var inline = type == 'latex-inline';
        var data = $(this).html();
        var new_elem = document.createElement(inline ? "span" : "div");
        $(new_elem).addClass('latex');
        $(this).replaceWith(new_elem);
		katex.render(data, new_elem, {
		    throwOnError: false
		});
    });
});