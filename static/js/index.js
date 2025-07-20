// Theme toggle logic
$(document).ready(function(){
  // Restore saved theme from localStorage
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    $('html').attr('data-theme', savedTheme);
  }

  // Helper to update theme icon and table class
  function updateThemeUI(theme) {
    const $icon = $('#theme-icon');
    if (theme === 'dark') {
      $icon.removeClass('bi-sun-fill').addClass('bi-moon-fill');
      $('#probabilities-table').addClass('table-dark');
    } else {
      $icon.removeClass('bi-moon-fill').addClass('bi-sun-fill');
      $('#probabilities-table').removeClass('table-dark');
    }
  }

  // On page load, set icon and table class
  const initialTheme = $('html').attr('data-theme') || 'light';
  updateThemeUI(initialTheme);

  // Theme toggle button click handler
  $('#theme-toggle').click(() => {
    const currentTheme = $('html').attr('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    $('html').attr('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeUI(newTheme);
  });

  // File upload validation
  const $uploadForm = $('#upload-form');
  const $uploadError = $('#upload-error');
  const $fileInput = $('#fileInput');

  // Image preview logic
  $fileInput.on('change', function() {
    if (this.files.length > 0) {
      const file = this.files[0];
      const fileSize = file.size / 1024 / 1024; // in MB
      if (fileSize > 2) {
        $uploadError.text('Image size exceeds 2MB');
        this.value = '';
        // Optionally reset preview to default
        $('#image-preview').html("<i class='bx bxs-cloud-upload icon'></i><h3>Upload Image</h3><p>Image size must be less than 2MB</p>");
      } else {
        $uploadError.text('');
        const reader = new FileReader();
        reader.onload = function(e) {
          $('#image-preview').html('<img src="' + e.target.result + '" alt="Uploaded Eye Image" class="img-fluid">');
        };
        reader.readAsDataURL(file);
      }
    }
  });

  $uploadForm.on('submit', function(e) {
    if ($fileInput[0].files.length === 0 || $uploadError.text() !== '') {
      e.preventDefault();
      $uploadError.text('Please select a valid image before submitting.');
    }
  });
}); 