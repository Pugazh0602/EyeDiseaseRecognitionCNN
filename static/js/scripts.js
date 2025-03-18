// /static/js/scripts.js

document.addEventListener('DOMContentLoaded', function () {
    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    function setTheme() {
        if (localStorage.getItem('theme') === 'dark') {
            body.classList.add('dark-mode');
            themeToggle.innerHTML = '<i class="bi bi-sun-fill"></i>';
        } else {
            body.classList.remove('dark-mode');
            themeToggle.innerHTML = '<i class="bi bi-moon-fill"></i>';
        }
    }

    setTheme();

    themeToggle.addEventListener('click', () => {
        if (body.classList.contains('dark-mode')) {
            body.classList.remove('dark-mode');
            themeToggle.innerHTML = '<i class="bi bi-moon-fill"></i>';
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.add('dark-mode');
            themeToggle.innerHTML = '<i class="bi bi-sun-fill"></i>';
            localStorage.setItem('theme', 'dark');
        }
    });

    // Image Upload Preview
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('image-preview');

    fileInput.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                imagePreview.innerHTML = `<img src="`+e.target.result+`" alt="Uploaded Image" class="img-fluid">`;
            }

            reader.readAsDataURL(file);
        }
    });
});

/*
function sortTable(columnIndex) {
    const table = document.getElementById("probabilities-table");
    let rows, switching, i, x, y, shouldSwitch;
    switching = true;

    while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
            shouldSwitch = false;
            x = rows[i].getElementsByTagName("TD")[columnIndex];
            y = rows[i + 1].getElementsByTagName("TD")[columnIndex];
            if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                shouldSwitch = true;
                break;
            }
        }
        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
        }
    }
}
*/